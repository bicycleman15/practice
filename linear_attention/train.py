# Run using: 
# python -m linear_attention.train

import torch
from dataclasses import dataclass
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset.utils import tiny_stories_dataset
from linear_attention.model import LinearAttentionConfig, LinearAttentionLM, generate

@dataclass
class TrainConfig:

    batch_size = 8
    block_size = 128

    steps = 1000

    grad_accum = 1

    eval_after = 50

    tokenizer_name = "meta-llama/Llama-2-7b-hf"

def main():

    device = "mps"

    train_config = TrainConfig()
    
    config = LinearAttentionConfig()
    model = LinearAttentionLM(config)
    model.to(device)

    print(model)

    dataset = tiny_stories_dataset(
        train_config.block_size,
        train_config.batch_size,
    )
    train_iterator = iter(dataset)
    tokenizer = AutoTokenizer.from_pretrained(train_config.tokenizer_name)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    bar = tqdm(train_iterator, total=train_config.steps, desc="Training")

    # just making sure we don't have any gradients apriori
    optimizer.zero_grad()

    # amp stuff
    grad_scaler = torch.amp.GradScaler()

    for i, (x, y) in enumerate(bar):
        if i >= train_config.steps:
            break

        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(x)

            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                y.view(-1)
            )

        # loss.backward()
        grad_scaler.scale(loss).backward()

        if (i + 1) % train_config.grad_accum == 0:
            # optimizer.step()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            optimizer.zero_grad()

            bar.set_postfix_str(f"loss: {loss.item():02f}")

        if (i > 0) and (i % train_config.eval_after == 0):

            # let's generate some new tokens
            prompt = x[:1, :10] # [1, 10]

            # generate 30 new tokens; [1, 30]
            decoded_tokens = generate(model, prompt, max_new_tokens=30, temperature=0.8, top_k=50, top_p=0.95)

            print("Input:", tokenizer.decode(prompt[0]))
            print("Output:", tokenizer.decode(decoded_tokens[0]))
            print()


if __name__ == "__main__":
    main()