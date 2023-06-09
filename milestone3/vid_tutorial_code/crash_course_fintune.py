# 1 Prepate dataset
# 2 Load pretrained Tokenizer, call it with dataset -> encoding
# 3 Build PyTorch Dataset with encodings
# 4 Load pretrained model
# 5 a) Load Trainer and train it
#   b) or use native Pytorch training pipeline
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments

model_name = "distilbert-base-uncased"

def read_imdb_split(split_dir): # helper function to get text and label
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        thres = 0
        for text_file in (split_dir/label_dir).iterdir():
            if thres < 100:
                f = open(text_file, encoding='utf8')
                texts.append(f.read())
                labels.append(0 if label_dir == "neg" else 1)
                thres += 1

    return texts, labels

train_texts, train_labels = read_imdb_split("aclImdb/train")
test_texts, test_labels = read_imdb_split("aclImdb/test")

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)


class IMDBDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)
    
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_dataset = IMDBDataset(train_encodings, train_labels)
val_dataset = IMDBDataset(val_encodings, val_labels)
test_dataset = IMDBDataset(test_encodings, test_labels)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10
)

model = DistilBertForSequenceClassification.from_pretrained(model_name)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train() 



