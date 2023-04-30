import numpy as np
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm


# Global var
TEST_SIZE = 2000
FINE_TUNED_MODEL = "andyqin18/finetuned-bert-uncased"


# Define analyze function
def analyze(text: str):
    '''
    Input: Text string
    Output: Prediction array (6x1) with threshold prob > 0.5
    '''
    encoding = tokenizer(text, return_tensors="pt")
    encoding = {k: v.to(model.device) for k,v in encoding.items()}
    outputs = model(**encoding)
    logits = outputs.logits
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.5)] = 1
    return predictions


# Read dataset and randomly select testing texts and respective labels
df = pd.read_csv("milestone3/comp/train.csv")
labels = df.columns[2:]
num_label = len(labels)
train_texts = df["comment_text"].values
train_labels = df[labels].values

np.random.seed(1)
small_test_texts = np.random.choice(train_texts, size=TEST_SIZE, replace=False)

np.random.seed(1)
small_test_labels_idx = np.random.choice(train_labels.shape[0], size=TEST_SIZE, replace=False)
small_test_labels = train_labels[small_test_labels_idx, :]


# Load model and tokenizer. Prepare for analysis loop
model = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_MODEL)
tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL)
total_true = 0
total_success = 0
TP, FP, TN, FN = 0, 0, 0, 0


# Analysis Loop
for comment_idx in tqdm(range(TEST_SIZE), desc="Analyzing..."):
    comment = small_test_texts[comment_idx]
    target = small_test_labels[comment_idx]
    result = analyze(comment[:500])

    # Counting TP, FP, TN, FN
    for i in range(num_label):
        if result[i] == target[i]:
            if result[i] == 1:
                TP += 1
            else:
                TN += 1
        else:
            if result[i] == 1:
                FP += 1
            else:
                FN += 1

    # Counting success prediction of 1) each label, 2) label array 
    num_true = (result == target).sum()
    if num_true == len(labels):
        total_success += 1
    total_true += num_true

# Calculate performance
performance = {}
performance["label_accuracy"] = total_true/(len(labels) * TEST_SIZE)  # Success prediction of each label
performance["prediction_accuracy"] = total_success/TEST_SIZE  # Success prediction of all 6 labels for 1 sample
performance["precision"] = TP / (TP + FP)  # Label precision
performance["recall"] = TP / (TP + FN)  # Label recall
print(performance)