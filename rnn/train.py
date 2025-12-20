import torch
from dataclasses import dataclass
from tqdm import tqdm

from dataset.utils import tiny_stories_dataset
from rnn.model import RNNConfig, RNN

@dataclass
class TrainConfig:

    batch_size = 8
    block_size = 128

    steps = 1000

def main():

    device = "mps"

    train_config = TrainConfig()
    
    config = RNNConfig()
    model = RNN(config)
    model.to(device)

    print(model)

    dataset = tiny_stories_dataset(
        train_config.block_size,
        train_config.batch_size,
    )
    train_iterator = iter(dataset)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    bar = tqdm(train_iterator, total=train_config.steps, desc="Training")
    for i, (x, y) in enumerate(bar):
        optimizer.zero_grad()

        x, y = x.to(device), y.to(device)

        logits = model(x)

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            y.view(-1)
        )

        loss.backward()
        optimizer.step()

        bar.set_postfix_str(f"loss: {loss.item():02f}")

if __name__ == "__main__":
    main()