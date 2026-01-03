"""
Data augmentation

Signature imposée :
get_augmentation_transforms(config: dict) -> objet/transform callable (ou None)
"""

# NLP projects typically do not use augmentation by default.
# But the project template requires this file.

def apply_augmentation(text, config):
    # No augmentation for AG News
    return text
