MODEL_NAME = 'liuhaotian/llava-13b'
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
GROUP_SIZE = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
