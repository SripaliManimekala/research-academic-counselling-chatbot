import pandas as pd
import numpy as np
import re
import os

from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Set the path to the data folder
data_folder = os.path.join('..', 'data')  # Assuming your current working directory is 'bot'

# Set the path to the text file
text_file_path = os.path.join(data_folder, '1.txt')

# Read the content of the text file
with open(text_file_path, 'r') as file:
    text_data = file.read()

# Now 'text_data' contains the content of the file
# You can use 'text_data' for training your chatbot
print(text_data)

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # # Remove punctuation
    # text = re.sub(r'[^\w\s]', '', text)

    # Standardize contractions
    text = re.sub(r"can't", "cannot", text)

    # # Remove extra whitespaces
    # text = re.sub(r'\s+', ' ', text).strip()

    # # Remove excess newline characters
    # text = re.sub(r'\n+', '\n', text).strip()

    return text

text_data = preprocess_text(text_data)

def load_dataset(file_path, tokenizer, block_size = 128):
    dataset = TextDataset(
        tokenizer = tokenizer,
        file_path = file_path,
        block_size = block_size,
    )
    return dataset

def load_data_collator(tokenizer, mlm = False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm,
    )
    return data_collator

def train(train_file_path,model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps):
  tokenizer = GPT2Tokenizer.from_pretrained(model_name)
  train_dataset = load_dataset(train_file_path, tokenizer)
  data_collator = load_data_collator(tokenizer)

  tokenizer.save_pretrained(output_dir)

  model = GPT2LMHeadModel.from_pretrained(model_name)

  model.save_pretrained(output_dir)

  training_args = TrainingArguments(
          output_dir=output_dir,
          overwrite_output_dir=overwrite_output_dir,
          per_device_train_batch_size=per_device_train_batch_size,
          num_train_epochs=num_train_epochs,
      )

  trainer = Trainer(
          model=model,
          args=training_args,
          data_collator=data_collator,
          train_dataset=train_dataset,
  )

  trainer.train()
  trainer.save_model()

  # Model and Training Parameters
train_file_path = "/content/drive/MyDrive/Colab Notebooks/data/train.txt"
model_name = 'gpt2'
output_dir = '/content/drive/MyDrive/Colab Notebooks/models/chat_models/model1'
overwrite_output_dir = False
per_device_train_batch_size = 8
num_train_epochs = 10.0
save_steps = 50000