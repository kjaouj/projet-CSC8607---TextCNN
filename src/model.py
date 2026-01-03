"""
Construction du modèle (à implémenter par l'étudiant·e).

Signature imposée :
build_model(config: dict) -> torch.nn.Module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_filters,
        kernel_sizes,
        num_classes,
        dropout,
        pad_idx,
        use_batch_norm=True,
        embedding_matrix=None
    ):
        super().__init__()

        # --- Embedding layer ---
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # Load pretrained embeddings if provided
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
        # else: keep random initialisation

        # --- Convolution layers ---
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k)
            for k in kernel_sizes
        ])

        # --- BatchNorm (optional) ---
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(num_filters) if use_batch_norm else nn.Identity()
            for _ in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)

        # Final classifier
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)              # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)             # (batch, embed_dim, seq_len)

        convs = [F.relu(bn(conv(x))) for conv, bn in zip(self.convs, self.bns)]
        pooled = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in convs]

        out = torch.cat(pooled, dim=1)
        out = self.dropout(out)

        return self.fc(out)


def build_model(config: dict, meta: dict):
    """
    Builds the TextCNN model from configuration + metadata.
    """

    # Safety check
    model_type = config["model"].get("type", "").lower()
    assert model_type == "textcnn", "Only 'textcnn' is supported."

    # Fixed kernel sizes for this architecture
    kernel_sizes = config["model"].get("kernel_sizes", [3, 4, 5])
    embed_dim = config["model"].get("embed_dim", 100)
    num_filters = config["model"].get("num_filters", 100)

    return TextCNN(
        vocab_size=len(meta["vocab"]),
        embed_dim=embed_dim,
        num_filters=num_filters,
        kernel_sizes=kernel_sizes,
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
        pad_idx=meta["pad_idx"],
        use_batch_norm=config["model"].get("batch_norm", True),
        embedding_matrix=meta.get("embedding_matrix", None)   # <-- FIXED HERE
    )
