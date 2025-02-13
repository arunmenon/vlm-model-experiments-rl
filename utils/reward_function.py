import torch

def compute_reward(predictions, ground_truths):
    rewards = []
    for pred, truth in zip(predictions, ground_truths):
        if truth == "Offensive":
            rewards.append(1.0 if pred == "Offensive" else -1.0)  # True Positive or False Negative
        else:
            rewards.append(-0.2 if pred == "Offensive" else 0.1)  # False Positive or True Negative
    return torch.tensor(rewards)
