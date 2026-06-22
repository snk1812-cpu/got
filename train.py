from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from model import TinyGPT


DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/"
    "char-rnn/master/data/tinyshakespeare/input.txt"
)


class NextTokenDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[index : index + self.block_size]
        y = self.data[index + 1 : index + self.block_size + 1]
        return x, y


def download_data(path: Path) -> None:
    if path.exists():
        return

    print(f"Downloading dataset to {path} ...")
    urllib.request.urlretrieve(DATA_URL, path)


def build_vocabulary(text: str):
    characters = sorted(set(text))
    stoi = {character: index for index, character in enumerate(characters)}
    itos = {index: character for character, index in stoi.items()}
    return characters, stoi, itos


@torch.no_grad()
def estimate_loss(
    model: TinyGPT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    eval_batches: int = 20,
):
    model.eval()
    result = {}

    for split_name, loader in [("train", train_loader), ("validation", val_loader)]:
        losses = []
        for batch_index, (x, y) in enumerate(loader):
            if batch_index >= eval_batches:
                break

            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())

        result[split_name] = sum(losses) / max(len(losses), 1)

    model.train()
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Train a small GPT-2-style model.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps-per-epoch", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(42)
    random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    data_path = Path("input.txt")
    download_data(data_path)
    text = data_path.read_text(encoding="utf-8")

    characters, stoi, itos = build_vocabulary(text)
    encoded = torch.tensor([stoi[character] for character in text], dtype=torch.long)

    split_index = int(0.9 * len(encoded))
    train_data = encoded[:split_index]
    validation_data = encoded[split_index:]

    train_dataset = NextTokenDataset(train_data, args.block_size)
    validation_dataset = NextTokenDataset(validation_data, args.block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
    )

    model = TinyGPT(
        vocab_size=len(characters),
        block_size=args.block_size,
        emb_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    number_of_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    print(f"parameters: {number_of_parameters:,}")

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        completed_steps = 0

        for step, (x, y) in enumerate(train_loader, start=1):
            if step > args.max_steps_per_epoch:
                break

            x = x.to(device)
            y = y.to(device)

            _, loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            completed_steps += 1

        losses = estimate_loss(
            model=model,
            train_loader=train_loader,
            val_loader=validation_loader,
            device=device,
        )

        row = {
            "epoch": epoch,
            "training_batch_loss": running_loss / max(completed_steps, 1),
            "train_loss": losses["train"],
            "validation_loss": losses["validation"],
        }
        history.append(row)

        print(
            f"epoch {epoch:02d} | "
            f"train loss {row['train_loss']:.4f} | "
            f"validation loss {row['validation_loss']:.4f}"
        )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": len(characters),
            "block_size": args.block_size,
            "emb_dim": args.embedding_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
        },
        "stoi": stoi,
        "itos": {str(index): character for index, character in itos.items()},
    }

    torch.save(checkpoint, "tiny_gpt_checkpoint.pt")
    Path("training_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    print("Saved tiny_gpt_checkpoint.pt")
    print("Saved training_history.json")


if __name__ == "__main__":
    main()
