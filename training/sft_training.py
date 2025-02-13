from models.llava_model import load_model
from dataset.process_data import load_data
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler

def train_sft():
    model, processor = load_model()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    # Training loop placeholder
    print("Starting Supervised Fine-Tuning...")
