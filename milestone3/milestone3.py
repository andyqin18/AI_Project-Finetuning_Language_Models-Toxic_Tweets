from datasets import load_dataset

dataset = load_dataset("comp")
dataset["train"][100]