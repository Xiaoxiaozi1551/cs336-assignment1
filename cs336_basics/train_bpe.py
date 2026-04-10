import multiprocessing
import os

import regex as re
from collections import Counter
from typing import BinaryIO
import pickle


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # 获取文件的字节数大小
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks # 每个块的平均大小

    # 块边界位置的初始猜测，均匀分布
    # 块从前一个索引开始，不包括最后一个索引:[index, next_index)
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096 # 每次向前读取4KB

    for bi in range(1, len(chunk_boundaries) - 1): # 忽略第一个和最后一个索引，第一个是0，最后一个是结束
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position) # 初始化文件指针位置
        while True:
            mini_chunk = file.read(mini_chunk_size) # 从当前位置读4KB到某字节

            # 如果读取到空字节串，说明到达文件末尾（EOF）
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 在mini chunk中找出特殊token
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # 保证所有边界不同，且分块数量比desired_num_chunks少
    return sorted(set(chunk_boundaries))


# 对文本进行初次分词并统计词频
def pre_tokenization(
        text: str,
        special_tokens: list[str],
        pat: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
) -> Counter[str]:
    # 需要将特殊token作为分割界限，先对这些token转义，然后形成pattern
    special_token_pat = '|'.join([re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)])
    results = re.split(special_token_pat, text) # 对文本进行分割，得到列表
    word_counts = Counter[str]()
    # 还要对分割出的词进行分割
    for result in results:
        for match in re.finditer(pat, result):
            word_counts[match.group(0)] += 1

    return word_counts


def train_bpe(input_path: str,
              vocab_size: int,
              special_tokens: list[str],
              ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    input_path: 文件路径
    vocab_size: 词表大小
    special_tokens: 特殊 token
    """
    # 构建初始词表，包括特殊词和0-255的字节
    vocab: list[bytes] = [
        *[tok.encode("utf-8") for tok in special_tokens],
        *[bytes([i]) for i in range(256)],
    ]
    word_counts: Counter[str] = Counter() # 统计词频

    # 读取文件,Pre_tokenization
    num_processes = 8
    with multiprocessing.Manager() as manager:
        results = manager.list() # 存储每个进程的结果
        processes: list[multiprocessing.Process] = [] # 存储每个进程
        with open(input_path, 'rb') as f:
            boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk_bytes = f.read(end - start)
                # 规范化换行符：将 \r\n 和 \r 都转换为 \n
                chunk_bytes = chunk_bytes.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
                chunk = chunk_bytes.decode("utf-8", errors="replace")  # 使用 errors="replace" 处理解码错误
                # 设置worker函数，对每个chunk进行预分词
                def worker(res_list, text, spec_tok):
                    res = pre_tokenization(text, spec_tok)
                    res_list.append(res)
                p = multiprocessing.Process(
                    target = worker,
                    args = (results, chunk, special_tokens)
                )
                processes.append(p)
                p.start()
        for p in processes:
            p.join()
        for res in results:
            word_counts += res

    # 接下来要进行BPE Merge
    merges: list[tuple[bytes, bytes]] = [] # 存储每次的合并规则
    pair_counts = Counter[tuple[bytes, bytes]]() # 用于统计字节对的频率，如(b'l', b'o')
    pair2words: dict[tuple[bytes,bytes], Counter[str]] = ({}) # 反向映射：字节对 -> 包含该对的单词及出现次数
    words_info: dict[str, list[bytes] | int] = ({}) # 存储每个分词的信息，包括分词状态和词频
    # 初始化BPE
    for w, count in word_counts.items():
        if w.encode("utf-8") in vocab: # 本身在词表中则跳过
            continue
        tokens = [bytes([t]) for t in w.encode("utf-8")] # 单词转为字节token
        word = {
            "tokens": tokens,
            "count": count
        } # 单个word的信息
        words_info[w] = word
        for i in range(len(tokens) - 1):
            pairs = (tokens[i], tokens[i+1])
            pair_counts[pairs] += count
            pair2words.setdefault(pairs, Counter())[w] += 1

    # 开始 BPE Merge
    while len(vocab) < vocab_size:
        # if len(vocab) == 265:
        #     print("=============================")
        #     print(pair_counts)
        max_freq = 0
        merge_pair = None
        for pair, count in pair_counts.items():
            if max_freq < count:
                max_freq = count
                merge_pair = pair
            elif max_freq == count:
                merge_pair = max(merge_pair, pair)
        """进行合并"""
        new_char = b''
        # print(merge_pair)
        # print(pair_counts)
        merges.append(merge_pair) # 记录一个即将合并的字节对
        # 对pair_counts的其他字节对频率进行更新
        for word, _ in pair2words[merge_pair].items(): # 先找到合并字节对的对应单词
            tok = words_info[word]["tokens"]
            count = words_info[word]["count"]
            i = 0
            while i < len(tok) - 1:
                if (tok[i], tok[i+1]) == merge_pair:
                    a = tok.pop(i)
                    b = tok.pop(i)
                    new_char = a + b # 合并的新字节
                    tok.insert(i, new_char)
                    words_info[word]["tokens"] = tok # 词的token信息更新
                    # 新字节对与后一个字节
                    if i + 1 < len(tok):
                        pair_counts[(b, tok[i+1])] -= count
                        pair2words[(b, tok[i+1])][word] -= 1
                        pair_counts[(new_char, tok[i+1])] += count # 新的字节对加入统计
                        pair2words.setdefault((new_char, tok[i + 1]), Counter())[word] += 1
                        if pair_counts[(b, tok[i+1])] == 0:
                            del pair_counts[(b, tok[i+1])]
                        if pair2words[(b, tok[i+1])][word] == 0:
                            del pair2words[(b, tok[i+1])][word]
                    # 新字节对与前一个字节
                    if i > 0:
                        pair_counts[(tok[i-1], a)] -= count
                        pair2words[(tok[i-1], a)][word] -= 1
                        pair_counts[(tok[i-1], new_char)] += count
                        pair2words.setdefault((tok[i-1], new_char), Counter())[word] += 1
                        if pair_counts[(tok[i-1], a)] == 0:
                            del pair_counts[(tok[i-1], a)]
                        if pair2words[(tok[i-1], a)][word] == 0:
                            del pair2words[(tok[i-1], a)][word]

                i += 1
        del pair_counts[merge_pair]
        # print(new_char)
        # break
        vocab.append(new_char)
    # print(vocab)

    return {i: tok for i, tok in enumerate(vocab)}, merges

if __name__ == "__main__":
    vocab, merges = train_bpe("../data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    with open("bpe_vocab.txt", "w", encoding="utf-8") as f:
        for i, tok in vocab.items():
            f.write(f"{i}\t{tok}\n")
    with open("bpe_merges.txt", "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{left} {right}\n")
    with open("bpe_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("bpe_merges.pkl", "wb") as f:
        pickle.dump(merges, f)