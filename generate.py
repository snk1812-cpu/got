from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import TinyGPT


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with the trained model.")
    parser.add_argument("--prompt", type=str, default="ROMEO:")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = Path("tiny_gpt_checkpoint.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "tiny_gpt_checkpoint.pt was not found. Run `python train.py` first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    stoi = checkpoint["stoi"]
    itos = {
        int(index): character
        for index, character in checkpoint["itos"].items()
    }

    model = TinyGPT(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt_indices = [
        stoi[character]
        for character in args.prompt
        if character in stoi
    ]

    if not prompt_indices:
        prompt_indices = [0]

    context = torch.tensor(
        [prompt_indices],
        dtype=torch.long,
        device=device,
    )

    generated = model.generate(
        context,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    generated_text = "".join(
        itos[int(index)]
        for index in generated[0].tolist()
    )

    print(generated_text)
    Path("generated_text.txt").write_text(
        generated_text,
        encoding="utf-8",
    )
    print("\nSaved generated_text.txt")


if __name__ == "__main__":
    main()
