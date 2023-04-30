---
title: Sentiment Analysis App
emoji: 🚀
colorFrom: green
colorTo: purple
sdk: streamlit
sdk_version: 1.17.0
app_file: app.py
pinned: false
---

# AI Project: Finetuning Language Models - Toxic Tweets

Hello! This is a project for CS-UY 4613: Artificial Intelligence. I'm providing a step-by-step instruction on finetuning language models for detecting toxic tweets. All codes are well commented.

# Everthing you need to know
Link to HuggingFace space: https://huggingface.co/spaces/andyqin18/sentiment-analysis-app

----Code behind app: [app.py](app.py)

Link to finetuned model: https://huggingface.co/andyqin18/finetuned-bert-uncased

----Code for how to finetune a language model: [finetune.ipynb](milestone3/finetune.ipynb)

Performance of the model using [test_model.py](test_model.py) is shown below. The result is generated on 2000 randomly selected samples from [train.csv](milestone3/comp/train.csv)

```
{'label_accuracy': 0.9821666666666666, 
 'prediction_accuracy': 0.9195, 
 'precision': 0.8263888888888888, 
 'recall': 0.719758064516129}
```

Now let's walk through the details :)

# Milestone 1 - Setup

This milestone includes setting up docker and creating a development environment on Windows 11.

## 1. Enable WSL2 feature

The Windows Subsystem for Linux (WSL) lets developers install a Linux distribution on Windows.

```
wsl --install 
```

Ubuntu is the default distribution installed and WSL2 is the default version. 
After creating linux username and password, Ubuntu can be seen in Windows Terminal now.
Details can be found [here](https://learn.microsoft.com/en-us/windows/wsl/install).

![](milestone1/wsl2.png)

## 2. Download and install the Linux kernel update package

The package needs to be downloaded before installing Docker Desktop. 
However, this error might occur:

`Error: wsl_update_x64.msi unable to run because "This update only applies to machines with the Windows Subsystem for Linux"`

Solution: Opened Windows features and enabled "Windows Subsystem for Linux". 
Successfully ran update [package](https://docs.microsoft.com/windows/wsl/wsl2-kernel).

![](milestone1/kernal_update_sol.png)

## 3. Download Docker Desktop

After downloading the [Docker App](https://www.docker.com/products/docker-desktop/), WSL2 based engine is automatically enabled. 
If not, follow [this link](https://docs.docker.com/desktop/windows/wsl/) for steps to turn on WSL2 backend. 
Open the app and input `docker version` in Terminal to check server running.

![](milestone1/docker_version.png)
Docker is ready to go.

## 4. Create project container and image

First we download the Ubuntu image from Docker’s library with:
```
docker pull ubuntu
```
We can check the available images with:
```
docker image ls
```
We can create a container named *AI_project* based on Ubuntu image with:
```
docker run -it --name=AI_project ubuntu
```
The `–it` options instruct the container to launch in interactive mode and enable a Terminal typing interface. 
After this, a shell is generated and we are directed to Linux Terminal within the container.
`root` represents the currently logged-in user with highest privileges, and `249cf37645b4` is the container ID.

![](milestone1/docker_create_container.png)

## 5. Hello World!

Now we can mess with the container by downloading python and pip needed for the project. 
First we update and upgrade packages by: (`apt` is Advanced Packaging Tool)
```
apt update && apt upgrade
```
Then we download python and pip with:
```
apt install python3 pip
```
We can confirm successful installation by checking the current version of python and pip.
Then create a script file of *hello_world.py* under `root` directory, and run the script.
You will see the following in VSCode and Terminal.

![](milestone1/vscode.png)
![](milestone1/hello_world.png)

## 6. Commit changes to a new image specifically for the project

After setting up the container we can commit changes to a specific project image with a tag of *milestone1* with:
```
docker commit [CONTAINER] [NEW_IMAGE]:[TAG]
```
Now if we check the available images there should be a new image for the project. If we list all containers we should be able to identify the one we were working on through container ID.

![](milestone1/commit_to_new_image.png)

The Docker Desktop app should match the image list we see on Terminal.

![](milestone1/app_image_list.png)

# Milestone 2 - Sentiment Analysis App w/ Pretrained Model

This milestone includes creating a Streamlit app in HuggingFace for sentiment analysis.

## 1. Space setup

After creating a HuggingFace account, we can create our app as a space and choose Streamlit as the space SDK.

![](milestone2/new_HF_space.png)

Then we can go back to our Github Repo and create the following files.
In order for the space to run properly, there must be at least three files in the root directory: 
[README.md](README.md), [app.py](app.py), and [requirements.txt](requirements.txt)

Make sure the following metadata is at the top of **README.md** for HuggingFace to identify.
```
---
title: Sentiment Analysis App
emoji: 🚀
colorFrom: green
colorTo: purple
sdk: streamlit
sdk_version: 1.17.0
app_file: app.py
pinned: false
---
```

The **app.py** file is the main code of the app and **requirements.txt** should include all the libraries the code uses. HuggingFace will install the libraries listed before running the virtual environment


## 2. Connect and sync to HuggingFace

Then we go to settings of the Github Repo and create a secret token to access the new HuggingFace space. 

![](milestone2/HF_token.png)
![](milestone2/github_token.png)

Next, we need to setup a workflow in Github Actions. Click "set up a workflow yourself" and replace all the code in `main.yaml` with the following: (Replace `HF_USERNAME` and `SPACE_NAME` with our own)

```
name: Sync to Hugging Face hub
on:
  push:
    branches: [main]

  # to run this workflow manually from the Actions tab
  workflow_dispatch:

jobs:
  sync-to-hub:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0
          lfs: true
      - name: Push to hub
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: git push --force https://HF_USERNAME:$HF_TOKEN@huggingface.co/spaces/HF_USERNAME/SPACE_NAME main
```
The Repo is now connected and synced with HuggingFace space!

## 3. Create the app

Modify [app.py](app.py) so that it takes in one text and generate an analysis using one of the provided models. Details are explained in comment lines. The app should look like this:

![](milestone2/app_UI.png)

# Milestone 3 - Finetuning Language Models

This milestone we wish to finetuning our own language model in HuggingFace for sentiment analysis.

Here's the setup block that includes all modules:
```
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
```

## 1. Prepare Data
First we extract comment strings and labels from `train.csv` and split them into training data and validation data with a percentage of 80% vs 20%. We also create 2 dictionaries that map labels to integers and back.
```
df = pd.read_csv("milestone3/comp/train.csv")

train_texts = df["comment_text"].values
labels = df.columns[2:]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
train_labels = df[labels].values


# Randomly select 20000 samples within the data
np.random.seed(18)
small_train_texts = np.random.choice(train_texts, size=20000, replace=False)

np.random.seed(18)
small_train_labels_idx = np.random.choice(train_labels.shape[0], size=20000, replace=False)
small_train_labels = train_labels[small_train_labels_idx, :]


# Separate training data and validation data
train_texts, val_texts, train_labels, val_labels = train_test_split(small_train_texts, small_train_labels, test_size=.2)
```

## 2. Data Preprocessing
As models like BERT don't expect text as direct input, but rather `input_ids`, etc., we tokenize the text using the tokenizer. The `AutoTokenizer` will automatically load the appropriate tokenizer based on the checkpoint on the hub. We can now merge the labels and texts to datasets as a class we defined.
```
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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


train_dataset = TextDataset(train_texts, train_labels)
val_dataset = TextDataset(val_texts, val_labels)
```

## 3. Train the model using Trainer
We define a model that includes a pre-trained base and also set the problem to `multi_label_classification`. Then we train the model using `Trainer`, which requires  `TrainingArguments` beforehand that specify training hyperparameters, where we can set learning rate, batch sizes and `push_to_hub=True`.

After verifying Token with HuggingFace, the model is now pushed to the hub.

```
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id)
model.to(device)

training_args = TrainingArguments(
    output_dir="finetuned-bert-uncased",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    load_best_model_at_end=True,
    push_to_hub=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

trainer.train()
trainer.push_to_hub()
```

## 4. Update the app

Modify [app.py](app.py) so that it takes in one text and generate an analysis using one of the provided models. Details are explained in comment lines. The app should look like this:

![](milestone3/appUI.png)

## Reference:
For connecting Github with HuggingFace, check this [video](https://www.youtube.com/watch?v=8hOzsFETm4I).

For creating the app, check this [video](https://www.youtube.com/watch?v=GSt00_-0ncQ)

The HuggingFace documentation is [here](https://huggingface.co/docs), and Streamlit APIs [here](https://docs.streamlit.io/library/api-reference).