"""
Chargement des données.

Signature imposée :
get_dataloaders(config: dict) -> (train_loader, val_loader, test_loader, meta: dict)

Le dictionnaire meta doit contenir au minimum :
- "num_classes": int
- "input_shape": tuple (ex: (3, 32, 32) pour des images)
"""

import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab as vocab_factory, GloVe
from collections import Counter
from torchtext.datasets import AG_NEWS


# -------------------------------------------------------------
#  SAFETY FIX: CONVERT ANY INPUT TO A CLEAN STRING
# -------------------------------------------------------------
def to_text(x):
    """Force AG_NEWS sample to be a clean string."""
    if isinstance(x, str):
        return x
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8", errors="ignore")
        except:
            return str(x)
    # If torch.Tensor or anything else → convert safely
    try:
        return str(x)
    except:
        return ""


# -------------------------------------------------------------
#  LABEL FIX
# -------------------------------------------------------------
def fix_label(raw_label):
    try:
        lab = int(raw_label) - 1
    except:
        lab = 0
    return max(0, min(3, lab))


# -------------------------------------------------------------
#  DATASET
# -------------------------------------------------------------
class AGNewsDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, vocab, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx])
        numerical = [self.vocab[token] for token in tokens]

        if len(numerical) < self.max_len:
            numerical = numerical + [0] * (self.max_len - len(numerical))
        else:
            numerical = numerical[:self.max_len]

        return torch.tensor(numerical, dtype=torch.long), self.labels[idx]


# -------------------------------------------------------------
#  COLLATE
# -------------------------------------------------------------
def collate_fn(batch):
    x = torch.stack([item[0] for item in batch])
    y = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return x, y


# -------------------------------------------------------------
#  MAIN DATALOADER FUNC
# -------------------------------------------------------------
def get_dataloaders(config):

    tokenizer = get_tokenizer("basic_english")

    print("Loading AG_NEWS (legacy torchtext 0.6.0 mode)...")
    train_iter, test_iter = AG_NEWS(root='.data')

    train_examples = list(train_iter)
    test_examples = list(test_iter)

    # Split 10%
    n_total = len(train_examples)
    n_val = int(0.1 * n_total)

    train_raw = train_examples[n_val:]
    val_raw = train_examples[:n_val]

    # ---- TEXT CONVERSION PATCH ----
    train_texts = [to_text(ex[1]) for ex in train_raw]
    val_texts   = [to_text(ex[1]) for ex in val_raw]
    test_texts  = [to_text(ex[1]) for ex in test_examples]

    train_labels = [fix_label(ex[0]) for ex in train_raw]
    val_labels   = [fix_label(ex[0]) for ex in val_raw]
    test_labels  = [fix_label(ex[0]) for ex in test_examples]

    print("Train labels range:", min(train_labels), max(train_labels))

    print("Building vocabulary...")
    counter = Counter()

    for text in train_texts:
        counter.update(tokenizer(text))

    # Sort by freq for consistent vocab building
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = dict(sorted_by_freq_tuples)
    
    # Modern torchtext vocab creation
    vocab = vocab_factory(ordered_dict, specials=["<pad>", "<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    print("Vocab size:", len(vocab))

    print("Loading GloVe vectors...")
    embed_dim = config.get("model", {}).get("embed_dim", 100)
    print(f"Target embedding dimension: {embed_dim}")
    
    # GloVe 6B supports 50, 100, 200, 300
    if embed_dim not in [50, 100, 200, 300]:
        print(f"Warning: GloVe 6B does not support dim={embed_dim}. Falling back to 100.")
        glove_dim = 100
    else:
        glove_dim = embed_dim

    glove = GloVe(name="6B", dim=glove_dim)
    
    # Manually build embedding matrix
    embedding_matrix = glove.get_vecs_by_tokens(vocab.get_itos())

    max_len = config["data"]["max_len"]
    batch_size = config["train"]["batch_size"]

    # ---- OVERFIT SMALL MODE ----
    # If overfit_small is True, we only use the first 1000 examples
    if config["train"].get("overfit_small", False):
        print("!!! OVERFIT SMALL MODE: Using only 1000 training examples !!!")
        train_texts = train_texts[:1000]
        train_labels = train_labels[:1000]

    train_dataset = AGNewsDataset(train_texts, train_labels, vocab, tokenizer, max_len)
    val_dataset = AGNewsDataset(val_texts, val_labels, vocab, tokenizer, max_len)
    test_dataset = AGNewsDataset(test_texts, test_labels, vocab, tokenizer, max_len)

    num_workers = config["dataset"].get("num_workers", 0)
    pin_memory = torch.cuda.is_available()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    meta = {
        "vocab": vocab,
        "tokenizer": tokenizer,
        "num_classes": 4,
        "embedding_dim": glove_dim,
        "pad_idx": vocab["<pad>"] if "<pad>" in vocab else 0, # Safety if no pad
        "unk_idx": vocab["<unk>"],
        "embedding_matrix": embedding_matrix
    }

    return train_loader, val_loader, test_loader, meta