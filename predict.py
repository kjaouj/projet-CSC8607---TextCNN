import torch
from src.utils import load_config, get_device
from src.data_loading import get_dataloaders
from src.model import build_model

def predict(text, model, vocab, tokenizer, max_len, device):
    model.eval()
    # 1. Tokenize
    tokens = tokenizer(text)
    # 2. Numericalize (using the specific vocab from training)
    numerical = [vocab[token] for token in tokens]
    
    # 3. Pad / Truncate
    if len(numerical) < max_len:
        numerical = numerical + [vocab["<pad>"]] * (max_len - len(numerical))
    else:
        numerical = numerical[:max_len]
        
    # 4. Predict
    tensor = torch.tensor([numerical], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(tensor)
        prediction = torch.argmax(logits, dim=1).item()
    
    classes = ["World", "Sports", "Business", "Sci/Tech"]
    return classes[prediction]

if __name__ == "__main__":
    config = load_config("configs/config.yaml")
    device = get_device(config)
    
    # Build everything to ensure vocab matches
    _, _, _, meta = get_dataloaders(config)
    model = build_model(config, meta).to(device)
    model.load_state_dict(torch.load("artifacts/best.ckpt", map_location=device))
    
    # Test with your own text!
    my_text = "The stock market reached new heights today as tech companies reported record profits."
    result = predict(my_text, model, meta['vocab'], meta['tokenizer'], config['data']['max_len'], device)
    
    print(f"\nText: {my_text}")
    print(f"Predicted Category: {result}")