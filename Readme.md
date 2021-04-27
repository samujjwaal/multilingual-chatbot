# Multilingual Chatbot

## Course Project - CS 521 Statistical NLP - Spring 2021

### Team Members: Meghana Jagadish and Samujjwaal Dey

## Overview

A chatbot for Pizza ordering service, as per user customization, supporting queries and the responses in more than one language.

Conventional Chatbots though are very useful, lack support for local languages. The goal of this project is to overcome that disadvantage and create a multilingual chatbot supporting at least 2-3 languages.

## Project Tree

```bash
.
├── Readme.md
├── Results.pdf
├── chatgui.py
├── models
│   ├── chatbot_model.h5
│   └── lid.176.ftz
├── requirements.txt
├── resources
│   ├── classes.pkl
│   ├── intents.json
│   ├── pizza.png
│   └── vocab.pkl
├── train_chatbot.py
└── utils
    ├── api_utils.py
    ├── chatbot_utils.py
    ├── fasttext_utils.py
    ├── transformer_utils.py
    └── translation_utils.py
```

## Application Description

|File|Description|
|---|---|
|`intents.json`|Domain specific intents dataset for chatbot|
|`vocab.pkl`|Serialized chatbot model vocabulary|
|`classes.pkl`|Serialized chatbot intent classes|
|`train_chatbot.py`|Script to train the chatbot from intents|
|`chatbot_model.h5`|Locally saved, trained chatbot model|
|`chatgui.py`|Script to create chatbot GUI|
|`chatbot_utils.py`|Define functions for using chatbot model|
|`translation_utils.py`|Define functions for text translation|
|`fasttext_utils.py`|Define functions for fasttext language detection|
|`lid.176.ftz`|Locally saved fasttext pre-trained model|
|`transformer_utils.py`|Define functions for loading transformer models|
|`api_utils.py`|Define functions to use Hugging Face Inference API|

## Instructions to Execute

Clone this repository from GitHub and open the root directory in the terminal.

### Install Requirements

The [required pip packages](./requirements.txt) for successfully executing the project are:

```text
numpy==1.20.1
wget==3.2
requests==2.25.1
googletrans==3.1.0a0
fasttext==0.9.2
nltk==3.6.1
Keras==2.4.3
transformers==4.5.1
Pillow==8.2.0
```

### Train Chatbot Model

```bash
python train_chatbot.py 
```

The intents are parsed from `intents.json` to generate the chatbot vocabulary and train a Keras sequential model for 200 epochs.

### Run Chatbot GUI

```bash
python chatgui.py
```

The GUI window for the chatbot is created.

Each user input and chatbot response (with source language and translation) are printed in the terminal too.

## Results

Results for execution of the Multilingual Chatbot for Pizza Ordering can be found in [this file](./Results.pdf).

The results file shows chatbot conversations for 4 case scenarios along with screenshot of GUI.
