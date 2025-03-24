import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

data_config = config["data"]
model_config = config["model"]
training_config = config["training"]

#step2 :Dynamic Data Loading

# Load the dataset based on the provided file path
df = pd.read_json(data_config["file_path"])

# Extract text and labels from the dataset
texts = df[data_config["text_column"]].tolist()
labels = df[data_config["label_column"]].tolist()

# Tokenize texts using configurable max sequence length
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(model_config["pretrained_model_name"])

encodings = tokenizer(texts, truncation=True, padding=True, max_length=data_config["max_seq_length"], return_tensors='pt')

#step 3: Custom Dataset Class


class CustomTextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create dataset object
dataset = CustomTextDataset(encodings, labels)

# step 4:  Model Setup and Training

# Initialize the model with num_labels from the configuration
model = BertForSequenceClassification.from_pretrained(model_config["pretrained_model_name"], num_labels=model_config["num_labels"])

# step 5: create Data loader

# Load the data into a DataLoader with configurable batch size
train_loader = DataLoader(dataset, batch_size=training_config["batch_size"], shuffle=True)



