from setuptools import setup, find_packages

setup(
    name='grpo_trainer_vlm',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "peft",
        "trl",
        "accelerate",
        "evaluate",
        "scikit-learn",
        "torchvision",
        "pandas",
        "numpy"
    ],
)
