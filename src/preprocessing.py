"""
Pré-traitements.

Signature imposée :
get_preprocess_transforms(config: dict) -> objet/transform callable
"""

from torchtext.data.utils import get_tokenizer

def get_text_tokenizer(name: str):
    if name == "basic_english":
        return get_tokenizer("basic_english")
    raise ValueError(f"Unsupported tokenizer: {name}")
