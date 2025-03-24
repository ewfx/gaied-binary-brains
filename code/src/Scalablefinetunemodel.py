import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification,Trainier,TrainingArguments
from torch.utils.data import DataLoader


#step 1: Modify the Dataset for Multi-Task Classification
label_map_request = {
    "Adjustment": 0,
    "AU Transfer": 1,
    "Closting Notice": 2,
    "Commitmentn Change":3,
    "Fee Payment":4,
    "Commitmentn Change":5,
    "Money Movement-inbound":6,
    "Money Movement-outbound":7
    # More Request Types...
}
label_map_sub_request = {
    "Reallocation Fees":0, 
"Amendement Fees":1,
"Reallocation Principal":2,
"Cashless Roll":3,
"Decrease":4,
"increase":5,
"Ongoing Fee":6,
" Letter of Credit Fee":7,
"Principal":8,
"Interest":9,
"Prinicipal+Interest":10,
"Foreign Currency":11,

    # More Sub-Request Types...
}


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
emails = df[data_config["text"]].tolist()
string_labels_request = df[data_config["request_type"]].tolist()
string_labels_sub_request=df[data_config["request_type"]].tolist()


# Convert string labels to integers
request_labels = [label_map_request[label] for label in string_labels_request]
sub_request_labels = [label_map_sub_request[label] for label in string_labels_sub_request]


# Tokenize texts using configurable max sequence length
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(model_config["pretrained_model_name"])

encodings = tokenizer(emails, truncation=True, padding=True, max_length=data_config["max_seq_length"], return_tensors='pt')

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
dataset = CustomTextDataset(encodings, request_labels,sub_request_labels)
train_dataset = tokenized_datasets['train']
eval_dataset = tokenized_datasets['test']

# step 4:  Model Setup and Training

# Initialize the model with num_labels from the configuration
model = BertForSequenceClassification.from_pretrained(model_config["pretrained_model_name"], num_labels=model_config["num_labels"])

print('Initialize train args trainer')
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

print('create trainer')

trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=eval_dataset            # evaluation dataset
)
print('Train the model')
#train the modal
trainer.train()

print('Evaluate the model')
# step 3: Evaluate the Model
trainer.evaluate()
#Step 4: Save the Fine-Tuned Model
print('saving the medel')
model.save_pretrained("../../model/financial_email_model")
tokenizer.save_pretrained("../../model/financial_email_model")
#Step 5: Make Predictions with the Fine-Tuned Model

model = BertForSequenceClassification.from_pretrained("../../model/financial_email_model")
tokenizer = BertTokenizer.from_pretrained("../../model/financial_email_model")



# step 5: create Data loader

# Load the data into a DataLoader with configurable batch size
train_loader = DataLoader(dataset, batch_size=training_config["batch_size"], shuffle=True)



