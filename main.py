import pandas as pd
from fastai.text.core import BaseTokenizer, Tokenizer

twitter_train_df = pd.read_csv('nlp_dataset.csv', on_bad_lines='skip')

# Print data set info
print("Dataset size:", len(twitter_train_df), '\n')
# twitter_train_df.info()

# Print distribution across all training languages
tweet_distribution = twitter_train_df.groupby('language').count()['text']

print(tweet_distribution.head())


twitter_train_df = twitter_train_df.drop('language', axis=1)
twitter_train_df = twitter_train_df.head(100)
twitter_train_df = twitter_train_df.round({'label': 0})
twitter_train_df['label'] = twitter_train_df['label'].astype('int')
twitter_train_df.to_csv("clear_data.csv")

##### Test


import pandas as pd
import datasets
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification,Trainer, TrainingArguments
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import os

class config:
    input_path = './'
    model_path = '/kaggle/input/roberta-base'
    model = 'roberta-base'

    learning_rate = 2e-5
    weight_decay = 0.01

    epochs = 5
    batch_size = 32


twitter_train_df = datasets.Dataset.from_csv(config.input_path + 'clear_data.csv')

model = RobertaForSequenceClassification.from_pretrained(config.model)
tokenizer = RobertaTokenizerFast.from_pretrained(config.model, max_length=512)


# define a function that will tokenize the model, and will return the relevant inputs for the model
def tokenization(batched_text):
    return tokenizer(batched_text['text'], padding=True, truncation=True)


print(twitter_train_df)
train_data = twitter_train_df.map(tokenization)

print(train_data)
train_data.set_format('torch', columns=['text', 'label'])
print(train_data)

# define accuracy metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    label_names=["1", "2", "3", "4", "5"],
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    disable_tqdm=False,
    load_best_model_at_end=True,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=8,
    fp16=True,
    logging_dir='./logs',
    dataloader_num_workers=8,
    run_name='roberta-classification',
)


# instantiate the trainer class and check for available devices
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

trainer.train()