import os
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import numpy as np
import torch
from lightning.fabric.strategies import DeepSpeedStrategy, XLAStrategy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_parrot.adapter import Parrot, Config, mark_only_adapter_as_trainable, adapter_state_from_state_dict
from lit_parrot.tokenizer import Tokenizer
from lit_parrot.utils import lazy_load, check_valid_checkpoint_dir

def generate_prompt(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    #if example["input"]:
   #     return (
    #        "Below is an instruction that describes a task, paired with an input that provides further context. "
    #        "Write a response that appropriately completes the request.\n\n"
    #        f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
    #    )
    #else:
    return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Response:"
        )

eval_interval = 600
save_interval = 1000
eval_iters = 100
log_interval = 1
devices = 4

# Hyperparameters
learning_rate = 9e-3
batch_size = 64 / devices
micro_batch_size = 4
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
epoch_size = 1613 # train dataset size
num_epochs = 5
max_iters = num_epochs * (epoch_size // micro_batch_size) // devices
weight_decay = 0.02
warmup_iters = 2 * (epoch_size // micro_batch_size) // devices  # 2 epochs

ds_config = {
    "train_micro_batch_size_per_gpu": micro_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "zero_optimization": {"stage": 2},
}


def setup(
    data_dir: Path = Path("data"),
    checkpoint_dir: Path = Path("checkpoints/tiiuae/falcon-40b"),
    out_dir: Path = Path("out/adapter/ASDiv"),
    precision: Optional[str] = None,
    tpu: bool = False,
):
    if precision is None:
        precision = "32-true" if tpu else "16-mixed"
    strategy = (
        "auto"
        if devices <= 1
        else XLAStrategy(sync_module_states=False) if tpu else DeepSpeedStrategy(config=ds_config)
    )
    # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
    fabric_devices = "auto" if (tpu and devices > 1) else devices
    fabric = L.Fabric(devices=fabric_devices, strategy=strategy, precision=precision)
    fabric.launch(main, data_dir, checkpoint_dir, out_dir)


def main(
    fabric: L.Fabric = None,
    data_dir: Path = Path("data"),
    checkpoint_dir: Path = Path("checkpoints/tiiuae/falcon-40b"),
    out_dir: Path = Path("out/adapter/ASDiv"),
):
    check_valid_checkpoint_dir(checkpoint_dir)
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets(data_dir=data_dir)

    config = Config.from_name(name=checkpoint_dir.name)
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
    print(type(fabric))
    
    model = Parrot(config)
    with lazy_load(checkpoint_path) as checkpoint:
        model.load_state_dict(checkpoint, strict=False)

    mark_only_adapter_as_trainable(model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fabric.print(f"Number of trainable parameters: {num_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model, optimizer = fabric.setup(model, optimizer)

    train_time = time.time()
    train(fabric, model, optimizer, train_data, val_data, checkpoint_dir, out_dir)
    print(f"Training time: {(time.time()-train_time):.2f}s")

    # Save the final checkpoint at the end of training
    save_path = out_dir / "lit_model_adapter_finetuned.pth"
    fabric.print(f"Saving adapter weights to {str(save_path)!r}")
    save_model_checkpoint(fabric, model, save_path)


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    checkpoint_dir: Path,
    out_dir: Path,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0

    tokenizer = Tokenizer(checkpoint_dir / "tokenizer.json", checkpoint_dir / "tokenizer_config.json")

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()
    for iter_num in range(max_iters):
        if step_count <= warmup_iters:
            # linear warmup
            lr = learning_rate * step_count / warmup_iters
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        t0 = time.time()

        input_ids, targets = get_batch(fabric, train_data)

        with fabric.no_backward_sync(model, enabled=((iter_num + 1) % gradient_accumulation_steps != 0)):
            logits = model(input_ids)
            loss = loss_fn(logits, targets)
            fabric.backward(loss / gradient_accumulation_steps)

        if (iter_num + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            if fabric.device.type == "xla":
                xm.mark_step()
            optimizer.zero_grad()
            step_count += 1

            if step_count % eval_interval == 0:
                val_loss = validate(fabric, model, val_data, tokenizer)
                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                fabric.barrier()

            if step_count % save_interval == 0:
                save_path = out_dir / f"iter-{iter_num:06d}.pth"
                fabric.print(f"Saving adapter weights to {str(save_path)!r}")
                # TODO: Provide a function/script to merge the adapter weights with pretrained weights
                save_model_checkpoint(fabric, model, save_path)
        else:
            if fabric.device.type == "xla":
                xm.mark_step()

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")
            with open("output.txt", "a") as f:
                print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms", file=f)



@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray, tokenizer: Tokenizer) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        losses[k] = loss.item()
    val_loss = losses.mean()

    # produce an example:
    instruction = "Write a grade 4 Multiplication question and corresponding equation to solve the problem."
    fabric.print(instruction)
    sample = {"instruction": instruction, "input": ""}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, device=model.device)
    output = generate(
        model,
        idx=encoded,
        max_returned_tokens=len(encoded) + 100,
        max_seq_length=model.config.block_size,
        temperature=0.8,
    )
    output = tokenizer.decode(output)
    fabric.print(output)

    model.train()
    return val_loss.item()






