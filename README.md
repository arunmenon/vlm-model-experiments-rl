# GRPO Trainer for Vision-Language Models

This repository implements a complete training pipeline that uses **Group Relative Policy Optimization (GRPO)** to fine-tune 
Vision-Language Models (VLMs) for high-recall offensive content detection. In this context, “offensive content” includes 
categories such as **Nudity** and **Weapons**. The primary objective is to maximize recall (catching as many offensive cases 
as possible) while accepting some false positives.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Detailed Context: RL, GRPO, and Reward Functions](#detailed-context-rl-grpo-and-reward-functions)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
  - [GRPO Fine-Tuning](#grpo-fine-tuning)
  - [Evaluation](#evaluation)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

Modern vision-language models like **LLaVA**, **BLIP-2**, and others have demonstrated impressive capabilities in 
understanding both visual and textual inputs. However, deploying these models in safety-critical applications (e.g., content 
moderation) requires additional fine-tuning to ensure that offensive content is reliably detected. This repository provides a 
full pipeline that:
- Uses **Supervised Fine-Tuning (SFT)** as a warm-up phase.
- Applies **GRPO** to further align the model with a high-recall objective using reinforcement learning (RL).
- Implements a custom reward function that heavily penalizes false negatives (missed offensive content) while applying a 
milder penalty to false positives.

## Features

- **End-to-End Pipeline:** Includes data processing, model loading, supervised fine-tuning, GRPO-based reinforcement learning, 
and evaluation.
- **Custom Reward Function:** Designed to prioritize recall, with asymmetric rewards to enforce cautious behavior.
- **Modular Code Structure:** Separated into clear directories for dataset processing, models, training scripts, evaluation, 
and utilities.
- **Multi-Modal Support:** Designed for vision-language tasks, supporting both image and text inputs.
- **Easy-to-Use Scripts:** Bash scripts to run the SFT and GRPO training stages.
- **Extensible:** Built using popular libraries such as PyTorch, Hugging Face Transformers, and TRL, making it easy to modify 
for different models or reward functions.

## Detailed Context: RL, GRPO, and Reward Functions

### Reinforcement Learning (RL) in Model Fine-Tuning
Reinforcement Learning (RL) is a framework where an agent learns to take actions in an environment so as to maximize 
cumulative reward. Unlike traditional supervised learning, where a model learns from static labeled examples, RL allows the 
model (or "agent") to learn from its interactions with an environment, receiving feedback in the form of rewards or penalties. 
In the context of fine-tuning large models:
- **RLHF (Reinforcement Learning from Human Feedback):** Has been successfully used to align language models with human values 
by rewarding desired behaviors and penalizing harmful ones.
- **Policy Optimization:** Methods like Proximal Policy Optimization (PPO) have been popular for these tasks.

### Group Relative Policy Optimization (GRPO)
**GRPO** is a variant of policy optimization that simplifies the RL training pipeline by eliminating the need for a separate 
value function. Its key aspects include:
- **Grouped Sampling:** For each input prompt, the model generates a group of candidate outputs. These multiple outputs are 
compared relative to one another.
- **Relative Reward Signal:** Instead of learning absolute rewards via a critic network, GRPO computes the average reward for 
a group of outputs and then adjusts the policy based on each output's advantage (i.e., its reward relative to the group 
average).
- **Stability and Simplicity:** By leveraging relative rewards, GRPO avoids some of the complexities and instabilities that 
can occur when learning a separate value function, making it particularly attractive for fine-tuning large, multi-modal 
models.

### Reward Function Design for High-Recall Scenarios
The reward function is at the heart of the RL process. It defines what the model should learn to optimize. For a 
**high-recall** system aimed at offensive content detection, the reward function should be designed with these principles:
- **Heavy Reward for True Positives (TP):** When the model correctly flags offensive content (e.g., nudity, weapons), assign a 
strong positive reward (e.g., `+1`). This motivates the model to catch as many offenses as possible.
- **Severe Penalty for False Negatives (FN):** Missing offensive content is unacceptable in high-recall applications. A strong 
negative reward (e.g., `-1` or lower) is assigned if the model fails to flag actual offensive content.
- **Mild Penalty for False Positives (FP):** Flagging safe content as offensive should incur only a mild penalty (e.g., 
`-0.2`). This allows the model to be cautious (and overflag) without being overly punished for a false alarm.
- **Optional Small Reward for True Negatives (TN):** Correctly identifying safe content can be rewarded slightly (e.g., 
`+0.1`) to encourage balanced behavior.

This asymmetric design ensures the model is biased toward detecting offensive content, thereby maximizing recall. In the GRPO 
framework, each candidate output is evaluated using this reward function, and the relative differences drive the model’s 
updates.

## Repository Structure


