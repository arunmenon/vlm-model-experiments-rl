from transformers import AutoProcessor, AutoModelForVision2Seq
from config import MODEL_NAME, DEVICE

def load_model():
    model = AutoModelForVision2Seq.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    return model, processor
