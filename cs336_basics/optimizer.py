from jaxtyping import Float, Int
from typing import Callable, Iterable, Optional
import torch
import torch.nn as nn
import math

def cross_entropy(
        pred_logits: Float[torch.Tensor, "batch_size vocab_size"],
        target: Int[torch.Tensor, "batch_size"]
) -> Float[torch.Tensor, ""]:
    x = pred_logits - pred_logits.max(dim=-1, keepdim=True).values
    x = x.exp().sum(dim=-1).log() - x[torch.arange(x.shape[0]), target]
    return x.mean()

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # 把该参数的当前状态拿出来
                t = state.get("t", 1) # 从状态中获取迭代次数，默认为0
                m = state.get("m", torch.zeros_like(p)) # 一阶动量
                v = state.get("v", torch.zeros_like(p)) # 二阶动量
                grad = p.grad.data
                # 更新参数
                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * grad**2
                lr_t = lr * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data

                # 存回新参数值
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss

def lr_cosine_schedule(
        t: int,
        max_lr: float,
        min_lr: float,
        T_w: int,
        T_c: int
):
    lr = 0
    if t < T_w:
        lr = t / T_w * max_lr
    elif t <= T_c:
        lr = min_lr + (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (
            max_lr - min_lr
        ) / 2
    else:
        lr = min_lr
    return lr

def gradient_clipping(params: Iterable[torch.nn.Parameter], M: float):
    eps = 1e-6
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            grad = p.grad.data
            total_norm += (grad**2).sum().item() # 计算总范数
    total_norm = total_norm ** (1.0 / 2)

    grad_clip = M / (total_norm + eps)
    if total_norm >= M:
        for p in params:
            if p.grad is not None:
                p.grad.data *= grad_clip
