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

Hello! This is a project for CS-UY 4613: Artificial Intelligence. I'm providing a step-by-step instruction on finetuning language models for detecting toxic tweets.

# Milestone 3

This milestone includes finetuning a language model in HuggingFace for sentiment analysis.

Link to app: https://huggingface.co/spaces/andyqin18/sentiment-analysis-app

## 1. Space setup
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


## Reference:
For connecting Github with HuggingFace, check this [video](https://www.youtube.com/watch?v=8hOzsFETm4I).

For creating the app, check this [video](https://www.youtube.com/watch?v=GSt00_-0ncQ)

The HuggingFace documentation is [here](https://huggingface.co/docs), and Streamlit APIs [here](https://docs.streamlit.io/library/api-reference).
