# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# import torch
# import torch.nn.functional as F

# model_name = "andyqin18/test-finetuned"

# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# res = classifier(["Fuck your mom", 
#            "Hope you don't hate it"])

# for result in res:
#     print(result)
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import numpy as np

df = pd.read_csv("comp/train.csv")

train_texts = df["comment_text"].values
train_labels = df[df.columns[2:]].values
# print(train_labels[0])

# np.random.seed(123)
# small_train_texts = np.random.choice(train_texts, size=1000, replace=False)
# small_train_labels_idx = np.random.choice(train_labels.shape[0], size=1000, replace=False)
# small_train_labels = train_labels[small_train_labels_idx, :]


# train_texts, val_texts, train_labels, val_labels = train_test_split(small_train_texts, small_train_labels, test_size=.2)
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

class TextDataset(Dataset):
  def __init__(self,texts,labels):
    self.texts = texts
    self.labels = labels

  def __getitem__(self,idx):
    encodings = tokenizer(self.texts[idx], truncation=True, padding="max_length")
    item = {key: torch.tensor(val) for key, val in encodings.items()}
    item['labels'] = torch.tensor(self.labels[idx],dtype=torch.float32)
    del encodings
    return item

  def __len__(self):
    return len(self.labels)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
train_dataset = TextDataset(train_texts,train_labels)
val_dataset = TextDataset(val_texts, val_labels)
# small_train_dataset = train_dataset.shuffle(seed=42).select(range(1000))
# small_val_dataset = val_dataset.shuffle(seed=42).select(range(1000))



model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6,  problem_type="multi_label_classification")
model.to(device)
training_args = TrainingArguments(
    output_dir="finetuned-bert-uncased", 
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=5e-4,
    weight_decay=0.01,
    evaluation_strategy="epoch", 
    push_to_hub=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()