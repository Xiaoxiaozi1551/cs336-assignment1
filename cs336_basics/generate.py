import torch

from tokenizer import Tokenizer
from module import TransformerLM, softmax

def decode(
        tokenizer: Tokenizer,
        lm: TransformerLM,
        prompt: str,
        stop_token: str,
        context_length: str,
        temperature: float,
        top_p: float,
        device=None
):
    inputs_ids = tokenizer.encode(prompt)
    inputs_ids = torch.tensor(inputs_ids, dtype=torch.long, device=device).unsqueeze(0)
    stop_token_id = tokenizer.encode(stop_token)

    with torch.no_grad():
        for _ in range(context_length - inputs_ids.shape[1]):
            pred = lm(inputs_ids)
            pred = pred[:, -1, :] / temperature # 取每个序列最后一个词的预测
            probs = softmax(pred, dim=-1)

            # top_p 采样
            if top_p < 1.0:
                probs_sorted, indices_sorted = probs.sort(descending=True)
                cumsum_probs = probs_sorted.cumsum(dim=-1)
                cutoff = cumsum_probs > top_p
                cutoff[:, 0] = False # 即使第一个token累积概率就超过阈值，也至少保留它
                probs_sorted[cutoff] = 0 # 把不需要的概率置为0
                probs_sorted /= probs_sorted.sum(dim=-1, keepdim=True)
                next_token = indices_sorted.gather(
                    -1, torch.multinomial(probs_sorted, 1)
                ) # 采样某个概率并返回原来的索引位置
            else:
                torch.multinomial(probs, 1)

            if next_token.item() == stop_token_id:
                break
            inputs_ids = torch.cat([inputs_ids, next_token], dim=1)

    # 对拼接好的token进行解码
    output = tokenizer.decode(inputs_ids[0].cpu().tolist())
    return output

if __name__ == "__main__":
    import os

    from train import TinyStoriesConfig

    # Load tokenizer and model (paths are examples)
    tokenizer = Tokenizer.from_files(
        "/data/swzhou/cs336/assignment1-basics/cs336_basics/bpe_vocab.pkl",
        "/data/swzhou/cs336/assignment1-basics/cs336_basics/bpe_merges.pkl",
        special_tokens=["<|endoftext|>"],
    )
    config = TinyStoriesConfig
    lm = TransformerLM(
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
    checkpoint = torch.load(
        "../data/checkpoints/tiny_stories/checkpoint_final.pt",
        map_location=config["device"],
    )
    lm.load_state_dict(checkpoint["model"])
    lm.eval()

    prompt = "Once upon a time"
    stop_token = "<|endoftext|>"
    generated_text = decode(
        tokenizer,
        lm,
        prompt,
        stop_token,
        context_length=200,
        temperature=1.0,
        top_p=0.9,
        device=config["device"],
    )
    print(generated_text)