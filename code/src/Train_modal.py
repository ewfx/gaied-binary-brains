from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize_function(examples):
    return tokenizer(examples['email_text'], padding="max_length", truncation=True)
# step 1 : Load the data (assuming you have a CSV file)
dataset = Dataset.from_pandas(your_dataframe)

# Tokenize the data
tokenized_datasets = dataset.map(tokenize_function, batched=True)
#Split your dataset into training and validation sets
tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = tokenized_datasets['train']
eval_dataset = tokenized_datasets['test']

#Step 2: Fine-Tune the BERT Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # Adjust num_labels
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

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=eval_dataset            # evaluation dataset
)
#train the modal
trainer.train()

# step 3: Evaluate the Model
trainer.evaluate()
#Step 4: Save the Fine-Tuned Model
model.save_pretrained("./financial_email_model")
tokenizer.save_pretrained("./financial_email_model")
#Step 5: Make Predictions with the Fine-Tuned Model

model = BertForSequenceClassification.from_pretrained("./financial_email_model")
tokenizer = BertTokenizer.from_pretrained("./financial_email_model")

# New email to classify
new_email = "Dear customer, your invoice of $5000 is now overdue."

# Tokenize the input text
inputs = tokenizer(new_email, return_tensors="pt", truncation=True, padding=True, max_length=512)

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)

# Get the predicted label
logits = outputs.logits
predicted_class = torch.argmax(logits, dim=-1).item()

# Print the predicted class
print(f"Predicted class: {predicted_class}")
