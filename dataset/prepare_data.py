import os
import argparse
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import datasets
from transformers import AutoTokenizer

BUFFER_SIZE = 10_000_000

_tokenizer = None

def tokenize(x: dict):
    text = x["text"]

    global _tokenizer
    if _tokenizer is None:
        # load the tokenizer for this process
        _tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids = _tokenizer.encode(text, add_special_tokens=False)

    # also attach EOS
    input_ids = [_tokenizer.eos_token_id] + input_ids

    return input_ids

def main(split="train"):
    
    ds = datasets.load_dataset("roneneldan/TinyStories")[split]

    buf = np.empty((BUFFER_SIZE,), dtype=np.uint32)

    idx = 0

    bar = tqdm(total=len(buf), desc="Tokenizing")

    with mp.Pool(processes=8) as pool:
        for ids in pool.imap(tokenize, ds, chunksize=16):

            if idx + len(ids) >= BUFFER_SIZE:
                bar.close()
                break

            buf[idx : idx + len(ids)] = np.asarray(ids)

            idx += len(ids)

            bar.update(len(ids))

    # dump the buf
    dataset_folder = "dataset/tiny_stories"
    os.makedirs(dataset_folder, exist_ok=True)

    np.save(os.path.join(dataset_folder, f"{split}.npy"), buf[:idx])

    print(f"Done tokenizing {split} split : {idx} tokens!!")


if __name__ == "__main__":
    # do both train and test
    main(split="train")
    main(split="validation")