import math

import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float, Int
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        初始化线性变换模块
        参数:
            in_features (int): 输入的最终维度
            out_features (int): 输出的最终维度
            device (torch.device | None): 参数存储设备，默认为None
            dtype (torch.dtype | None): 参数数据类型，默认为None
        """
        super().__init__()
        # 使用Parameter将矩阵加入模型参数中
        self.W: Float[torch.Tensor, "in_features, out_features"] = nn.Parameter(
            torch.empty((in_features, out_features), device=device, dtype=dtype)
        )
        # 初始化线性层权重
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.W, 0, std, -3 * std, 3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入应用线性变换"""
        output = einsum(x, self.W, "... d_in, d_in d_out -> ... d_out")
        return output


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        初始化嵌入模块
        参数:
            num_embeddings (int): 词表大小（词汇量）
            embedding_dim (int): 嵌入向量的维度（即d_model）
            device (torch.device | None): 参数存储设备，默认为None
            dtype (torch.dtype | None): 参数数据类型，默认为None
        """
        super().__init__()
        self.embed: Float[torch.Tensor, "num_embeddings embedding_dim"] = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        # 嵌入层初始化
        nn.init.trunc_normal_(self.embed, 0, 1, -3, 3)

    def forward(self, token_ids: Int[torch.Tensor, "..."]) -> torch.Tensor:
        """根据输入的 token ID 查找对应的嵌入向量"""
        return self.embed[token_ids]


class RMSnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        初始化RMSNorm模块
        参数:
            d_model (int): 模型的隐藏层维度
            eps (float): 数值稳定项，默认为1e-5
            device (torch.device | None): 参数存储设备，默认为None
            dtype (torch.dtype | None): 参数数据类型，默认为None
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # 可学习参数
        self.g: Float[torch.Tensor, "d_model"] = nn.Parameter(
            torch.ones((d_model), device=device, dtype=dtype)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """处理输入张量（形状为(batch_size, sequence_length, d_model)）
        并返回相同形状的张量
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # RMSnorm 计算
        rms = torch.sqrt(1 / self.d_model * x.pow(2).sum(dim=2) + self.eps)
        rms = rearrange(rms, "b s-> b s 1")
        result = einsum(x / rms, self.g, "... d, d  -> ... d")

        # 将结果转换回原始数据类型
        return result.to(in_dtype)


def silu(x: torch.Tensor):
    return x * F.sigmoid(x)


class SwitchLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2.forward(silu(self.W1.forward(x)) * self.W3.forward(x))


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        初始化RoPE模块并创建缓冲区(如需要)
        参数:
            theta (float): RoPE的Θ参数值
            d_k (int): 查询向量和键向量的维度
            max_seq_len (int): 输入的最大序列长度
            device (torch.device | None): 缓冲区存储设备，默认为None
        """
        super().__init__()
        thetas = einsum(
            torch.arange(max_seq_len),
            torch.pow(theta, -torch.arange(0, d_k, 2) / d_k),
            "pos, theta -> pos theta"
        )
        cos = torch.cos(thetas).to(device)
        sin = torch.sin(thetas).to(device)
        #
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        处理输入张量(形状为(..., seq_len, d_k))并返回相同形状的张量
        参数:
            x: 任意批次维度的输入张量
            token_positions: 形状为(..., seq_len)的位置张量，指定x在序列维度的位置
            使用token_positions参数对预计算的cos/sin张量进行切片

        """
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        # 取出sin与cos的对应切片
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        result = torch.empty_like(x) # 把位置编码的形状固定为输入向量的形状
        result[..., ::2] = x_even * cos - x_odd * sin
        result[..., 1::2] = x_odd * cos + x_even * sin
        return result

def softmax(x: Float[torch.Tensor, "..."], dim: int):
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x = torch.exp(x - x_max)
    return x / torch.sum(x, dim=dim, keepdim=True)

def scale_dot_product_attention(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys d_k"],
    V: Float[torch.Tensor, " ... values d_v"],
    mask: Float[torch.Tensor, " ... queries keys"] | None = None,
):
    d_k = Q.shape[-1]
    scores = einsum(
        Q, K,
        "... queries d_k, ... keys d_k -> ... queries keys"
    )

    if mask is not None:
       scores = scores.masked_fill(mask == False, value=float("-inf"))
    attn = einsum(
        softmax(scores / math.sqrt(d_k), -1), V,
        "... queries keys, ... keys d_v -> ... queries d_v"
    )
    return attn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, pos_encoder: nn.Module = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.pos_encoder = pos_encoder
        self.device = device
        self.dtype = dtype
        # 初始化K、Q、V权重矩阵
        self.W_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_o = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
            self,
            x: [torch.Tensor, "batch_size ... seq_len d_model"],
            token_positions: torch.Tensor | None = None
    ):
        seq_len = x.shape[-2]
        Q: Float[torch.Tensor, "batch_size ... seq_len d_model"] = self.W_q(x)
        K: Float[torch.Tensor, "batch_size ... seq_len d_model"] = self.W_k(x)
        V: Float[torch.Tensor, "batch_size ... seq_len d_model"] = self.W_v(x)

        # 多头分割
        Qh: Float[torch.Tensor, "batch_size ... h seq_len d_q"] = rearrange(
            Q, "... seq_len (h d) -> ... h seq_len d", h=self.num_heads
        )
        Kh: Float[torch.Tensor, "batch_size ... h seq_len d_q"] = rearrange(
            K, "... seq_len (h d) -> ... h seq_len d", h=self.num_heads
        )
        Vh: Float[torch.Tensor, "batch_size ... h seq_len d_q"] = rearrange(
            V, "... seq_len (h d) -> ... h seq_len d", h=self.num_heads
        )

        # 位置编码
        if self.pos_encoder is not None and token_positions is not None:
            Qh = self.pos_encoder(Qh, token_positions)
            Kh = self.pos_encoder(Kh, token_positions)
        # 注意力掩码：下三角矩阵
        mask = (
            torch.triu(
                torch.ones(seq_len, seq_len, device=self.device, dtype=torch.bool),
                diagonal=1
            ) == 0
        )
        # 注意力计算
        attn = scale_dot_product_attention(Qh, Kh, Vh, mask)
        attn = rearrange(
            attn, "... h seq_len d -> ... seq_len (h d)"
        )

        return self.W_o(attn)

class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            max_seq_len: int,
            theta: float,
            device=None,
            dtype=None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # 归一化层
        self.norm1 = RMSnorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSnorm(d_model, device=device, dtype=dtype)
        # 前馈网络
        self.swiglu = SwitchLU(d_model, d_ff, device=device, dtype=dtype)
        # 注意力机制
        self.rope = RoPE(theta, d_model // num_heads, max_seq_len, device)
        self.mha = MultiHeadSelfAttention(d_model, num_heads, self.rope, device, dtype)
        self.token_positions: Int[torch.Tensor, "max_seq_len"] = torch.arange(
            max_seq_len, device=device
        )

    def forward(self, x: torch.Tensor):
        x_residual = x
        x = self.norm1(x)
        x = self.mha(x, self.token_positions[: x.shape[-2]])
        x = x_residual + x

        x_residual = x
        x = self.norm2(x)
        x = self.swiglu(x)
        x = x_residual + x

        return x

class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float,
            device=None,
            dtype=None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.cantext_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.token_embed = Embedding(vocab_size, d_model, device, dtype)
        # Transformer 块堆叠
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device, dtype)
            for _ in range(num_layers)
        ])
        self.norm = RMSnorm(d_model, device=device, dtype=dtype)
        self.ffn = Linear(d_model, vocab_size, device, dtype)

    def forward(self, in_indices: Int[torch.Tensor, "batch_size seq_len"]):
        x = self.token_embed(in_indices)
        for block in self.layers:
            x = block(x)
        x = self.norm(x)
        x = self.ffn(x)
        return x