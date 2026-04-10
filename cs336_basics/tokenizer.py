import functools
import pickle
from typing import Iterable, Iterator

import regex as re


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        ## 构造函数，接收以下参数创建分词器：
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens: list[str] = special_tokens or []

        self.pat: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.compiled_token_pattern = re.compile(self.pat)
        self.special_tokens_pat = (("(" +
                                   "|".join([re.escape(f"{token}") for token in sorted(special_tokens, key=len, reverse=True)]))
                                   + ")") if self.special_tokens else "(?!)" # 需要保证特殊token不被分割
        self.compiled_special_tokens_pattern = re.compile(self.special_tokens_pat)

        self.token2id: dict[bytes: int] = {v: k for k, v in vocab.items()}
        self.merge2rank: dict[tuple[bytes, bytes], int] = {merge: i for i, merge in enumerate(self.merges)}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)\
            -> "Tokenizer":

        ## 类方法，从序列化的词汇表文件和合并记录文件（格式应与BPE训练代码输出一致）构造并返回Tokenizer实例，
        ## 接收参数：
        vocab_filepath: str = vocab_filepath
        merges_filepath: str = merges_filepath
        special_tokens: list[str] | None = special_tokens

        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    @functools.lru_cache(maxsize=16 * 1024)
    def encode(self, text: str) -> list[int]:
        """
        将输入文本编码为token ID序列
        """
        token_id: list[int] = []
        results = self.compiled_special_tokens_pattern.split(text) # 对文本初次分割，保留特殊token
        # print(results)
        for result in results:
            if result in self.special_tokens:
                token_id.append(self.token2id[result.encode("utf-8")])
                continue
            for match in self.compiled_token_pattern.finditer(result):
                word = match.group(0)
                # 处理word，找到分割
                token = [bytes([i]) for i in word.encode("utf-8")]

                # 判断merge优先级
                def get_merge(tok):
                    candidate = None
                    min_rank = float("inf")
                    for loc, t in enumerate(tok):
                        if loc == len(tok) - 1:
                            break
                        pairs = (tok[loc], tok[loc + 1])
                        if pairs in self.merge2rank and self.merge2rank[pairs] < min_rank:
                            min_rank = self.merge2rank[pairs]
                            candidate = pairs
                    return candidate

                while True:
                    cur_merge = get_merge(token)
                    if cur_merge is None:
                        break
                    i = 0
                    while i < len(token) - 1:
                        if token[i] == cur_merge[0] and token[i + 1] == cur_merge[1]:
                            a = token.pop(i)
                            b = token.pop(i)
                            token.insert(i, a+b)
                        i += 1

                encode_token: list[int] = []
                for i in token:
                    encode_token.append(self.token2id[i])

                token_id.extend(encode_token)

        return token_id


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:

    ## 接收字符串可迭代对象（如Python文件句柄），返回惰性生成token ID的生成器。
    ## 该方法用于高效处理无法直接加载到内存的大文件
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id


    def decode(self, ids: list[int]) -> str:
    ## 将token ID序列解码为文本
        bytes_list = [self.vocab[token_id] for token_id in ids]
        return b"".join(bytes_list).decode("utf-8", errors="replace")

def _encode_with_text(t: str):
    return (t, tokenizer.encode(t))

def _accumulate_iter(iterable: Iterable[str], min_size: int) -> Iterator[str]:
    """
    Accumulate strings from an iterable until reaching at least min_size,
    """
    batch = ""
    for text in iterable:
        batch += text
        if len(batch) >= min_size:
            yield batch
            batch = ""
    if batch:
        yield batch

def _init_worker(tok: Tokenizer):
    global tokenizer
    tokenizer = tok

if __name__ == "__main__":
    import array
    import multiprocessing
    import os

    import numpy as np
    import tqdm

    tokenizer = Tokenizer.from_files(
        "../cs336_basics/bpe_vocab.pkl",
        "../cs336_basics/bpe_merges.pkl",
        ["<|endoftext|>"]
    )

    # 测试简单单词
    test_str = "Hello"
    ids = tokenizer.encode(test_str)
    print(f"'{test_str}' -> {ids}")
    print(f"Decoded: {tokenizer.decode(ids)}")

    # 测试特殊token
    test_str = "Hello<|endoftext|>World"
    ids = tokenizer.encode(test_str)
    print(f"'{test_str}' -> {ids}")
    print(f"Decoded: {tokenizer.decode(ids)}")

    token_ids_buf = array.array("H")

    file_path = "../data/TinyStoriesV2-GPT4-train.txt"
    with open(file_path, "r") as f:
        f.seek(0, os.SEEK_END)
        bytes_len = f.tell()
        f.seek(0)

        with tqdm.tqdm(
                total=bytes_len,
                unit="char",
                desc="Encoding",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:,}/{total:,} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ) as pbar:
            with multiprocessing.Pool(
                    processes=8, initializer=_init_worker, initargs=(tokenizer,)
            ) as pool:
                batch_ids = pool.imap(
                    _encode_with_text, _accumulate_iter(f, 128 * 1024)
                )  # parallel processing
                for text, ids in batch_ids:
                    token_ids_buf.extend(ids)
                    pbar.update(len(text.encode("utf-8")))
    token_ids = np.frombuffer(token_ids_buf, dtype=np.uint16)
    np.save("token_ids.npy", token_ids)
    print(f"Compression ratio: {bytes_len / (token_ids.size):.2f}")
    exit()