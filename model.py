import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """One masked self-attention head."""

    def __init__(self, emb_dim: int, head_size: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, sequence_length, _ = x.shape

        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        attention = query @ key.transpose(-2, -1)
        attention = attention * (key.size(-1) ** -0.5)
        attention = attention.masked_fill(
            self.tril[:sequence_length, :sequence_length] == 0,
            float("-inf"),
        )
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        return attention @ value


class MultiHeadAttention(nn.Module):
    """Several attention heads executed in parallel."""

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        block_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        if emb_dim % num_heads != 0:
            raise ValueError("emb_dim must be divisible by num_heads")

        head_size = emb_dim // num_heads
        self.heads = nn.ModuleList(
            [
                Head(
                    emb_dim=emb_dim,
                    head_size=head_size,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(num_heads)
            ]
        )
        self.projection = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        output = self.projection(output)
        return self.dropout(output)


class FeedForward(nn.Module):
    """Position-wise feedforward neural network."""

    def __init__(self, emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerBlock(nn.Module):
    """Pre-layer-normalization Transformer decoder block."""

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        block_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(emb_dim)
        self.attention = MultiHeadAttention(
            emb_dim=emb_dim,
            num_heads=num_heads,
            block_size=block_size,
            dropout=dropout,
        )
        self.layer_norm_2 = nn.LayerNorm(emb_dim)
        self.feedforward = FeedForward(emb_dim=emb_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.feedforward(self.layer_norm_2(x))
        return x


class TinyGPT(nn.Module):
    """Small character-level GPT model based on decoder-only Transformers."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        emb_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(block_size, emb_dim)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(emb_dim)
        self.language_model_head = nn.Linear(emb_dim, vocab_size)

    def forward(
        self,
        indices: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, sequence_length = indices.shape

        if sequence_length > self.block_size:
            raise ValueError(
                f"sequence length {sequence_length} exceeds block_size {self.block_size}"
            )

        positions = torch.arange(sequence_length, device=indices.device)
        token_embeddings = self.token_embedding(indices)
        position_embeddings = self.position_embedding(positions)[None, :, :]

        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.final_layer_norm(x)
        logits = self.language_model_head(x)

        loss = None
        if targets is not None:
            batch_size, sequence_length, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.reshape(batch_size * sequence_length, vocab_size),
                targets.reshape(batch_size * sequence_length),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        indices: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()

        for _ in range(max_new_tokens):
            cropped = indices[:, -self.block_size :]
            logits, _ = self(cropped)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(
                    logits < values[:, [-1]],
                    float("-inf"),
                )

            probabilities = F.softmax(logits, dim=-1)
            next_index = torch.multinomial(probabilities, num_samples=1)
            indices = torch.cat([indices, next_index], dim=1)

        return indices
