import os
import typing

import numpy as np
import numpy.typing as npt
import torch
import wandb
from jaxtyping import Int
from module import TransformerLM
from optimizer import AdamW, cross_entropy, lr_cosine_schedule, gradient_clipping

os.environ["WANDB_MODE"] = "disabled"  # 完全禁用 wandb

def get_batch(x, batch_size, context_length, device=None):
    # 随机采样，选择每个样本的起始
    indices = np.random.randint(0, x.shape[0] - context_length, size=(batch_size, ))
    inputs = np.stack([x[i: i + context_length] for i in indices])
    targets = np.stack([x[i + 1: i + context_length + 1] for i in indices])
    inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
    targets_tensor = torch.tensor(targets, dtype=torch.long, device=device)
    return inputs_tensor, targets_tensor

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    ## 应将前三个参数的所有状态转储到文件类对象out中。
    ## 你可以使用模型和优化器的state_dict方法获取它们相关状态，
    ## 并使用torch.save(obj, out)将obj转储到out中（PyTorch支持路径或文件类对象）。
    ## 通常选择让obj成为一个字典，但只要之后能加载你的检查点，你可以使用任何格式。
    checkpoints = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoints, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    ## 应从src（路径或文件类对象）加载检查点，然后从该检查点恢复模型和优化器状态。
    ## 你的函数应返回保存到检查点的迭代次数。
    ## 你可以使用torch.load(src)恢复你在save_checkpoint实现中保存的内容，
    ##并使用模型和优化器中的load_state_dict方法将它们恢复到之前的状态。

    checkpoint = torch.load(src) # 返回模型、优化器、迭代次数的字典
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    iteration = checkpoint["iteration"]
    return iteration

# 训练参数配置
class TrainConfig(typing.TypedDict):
    # 设备与数据类型
    device: torch.device
    dtype: torch.dtype
    # TransformerLM
    vocab_size: int
    d_model: int
    context_length: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float
    # Optimizer
    lr: float
    lr_min: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    max_grad_norm: float
    # data
    token_id_path: str | os.PathLike
    checkpoint_dir: str | os.PathLike
    # train
    batch_size: int
    total_tokens: int
    validation_interval: int
    checkpoint_interval: int
    wandb_project: str
    wandb_name: str

TinyStoriesConfig = TrainConfig(
    device=torch.device("cuda"),
    dtype=torch.float32,
    # Transformer LM
    vocab_size=10000,
    context_length=256,
    d_model=512,
    num_layers=4,
    num_heads=16,
    d_ff=1344,
    rope_theta=10000,
    # optimizer
    lr=3e-4,
    lr_min=3e-5,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8,
    max_grad_norm=1.0,
    # data
    token_ids_path="/data/swzhou/cs336/assignment1-basics/cs336_basics/token_ids.npy",
    checkpoint_dir="../data/checkpoints/tiny_stories",
    # train
    batch_size=128,
    total_tokens=327_680_000,
    validation_interval=10,
    checkpoint_interval=1000,
    wandb_project="cs336",
    wandb_name="tiny_stories_a6000",
)

# 训练函数
def train(config: TrainConfig):
    # 创建model对象
    transformerlm = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        device=config["device"],
        dtype=config["dtype"],
    )
    # 创建optimizer对象
    adamw = AdamW(
        params=transformerlm.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        betas=config["betas"],
        eps=config["eps"]
    )

    token_ids = np.load(config["token_ids_path"], mmap_mode="r")
    total_steps = (
            config["total_tokens"] // config["batch_size"] // config["context_length"]
    )
    print(f"Total training steps: {total_steps}")
    wandb.init(
        project=config["wandb_project"],
        name=config["wandb_name"],
        config={**config, "total_steps": total_steps},
    )

    for step in range(total_steps):
        inputs, targets = get_batch(
            token_ids,
            config["batch_size"],
            config["context_length"],
            config["device"]
        )
        pred = transformerlm(inputs)
        loss = cross_entropy(pred.view(-1, pred.size(-1)), targets.view(-1))
        adamw.zero_grad() # 优化器初始化
        loss.backward() # 反向传播
        gradient_clipping(transformerlm.parameters(), config["max_grad_norm"]) # 梯度裁剪
        lr = lr_cosine_schedule(step, config["lr"], config["lr_min"], total_steps // 10, total_steps)
        for param_group in adamw.param_groups:
            param_group["lr"] = lr
        adamw.step() # 优化器更新参数

        wandb.log({"loss": loss.item(), "lr": lr}, step=step)
        if (step + 1) % config["validation_interval"] == 0:
            print(f"Step {step + 1}: loss = {loss.item():.4f}")
        if (step + 1) % config["checkpoint_interval"] == 0:
            os.makedirs(config["checkpoint_dir"], exist_ok=True)
            checkpoint_path = os.path.join(
                config["checkpoint_dir"], f"checkpoint_step_{step + 1}.pt"
            )
            save_checkpoint(transformerlm, adamw, step + 1, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    checkpoint_path = os.path.join(config["checkpoint_dir"], f"checkpoint_final.pt")
    save_checkpoint(
        transformerlm,
        adamw,
        total_steps,
        checkpoint_path,
    )


if __name__ == "__main__":
    train(TinyStoriesConfig)