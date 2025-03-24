from transformers import BertForSequenceClassification, BertTokenizer
import torch
import torch.nn as nn

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
#step 1: Modify the Dataset for Multi-Task Classification
label_map_sub_request = {
    "Home Loan": 0,
    "Personal Loan": 1,
    "Balance Inquiry": 2,
    "Details Request": 3,
    # More Sub-Request Types...
}

emails = [
    "I want to know the status of my home loan application.",
    "Can I check the balance in my account?"
]


string_labels_request = ["Adjustment","AU Transfer","Closing Notice","Commitment Change","Money Movement-inbound","Money Movement-outbound","Spam", "Loan", "Account", "Fee Payment", "Inquiry", "Other"]
string_labels_sub_request = ["Reallocation Fees", "Amendement Fees","Reallocation Principal","Cashless Roll","Decrease","increase","Ongoing Fee"," Letter of Credit Fee","Principal","Interest",
                             "Principal+Interest","Principal+Interest+Fee","Timebound","Foreign Currency","Spam", "Payment Issue", "Overpayment", "Balance Inquiry", "Account Query", "Other"]



# Convert string labels to integers
request_labels = [label_map_request[label] for label in string_labels_request]
sub_request_labels = [label_map_sub_request[label] for label in string_labels_sub_request]

#step 2:Model Setup for Multi-Task Classification
class BertForMultiTaskClassification(nn.Module):
    def __init__(self, pretrained_model_name, num_request_labels, num_sub_request_labels):
        super(BertForMultiTaskClassification, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.request_classifier = nn.Linear(self.bert.config.hidden_size, num_request_labels)
        self.sub_request_classifier = nn.Linear(self.bert.config.hidden_size, num_sub_request_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        # Request type output
        request_logits = self.request_classifier(pooled_output)
        # Sub-request type output
        sub_request_logits = self.sub_request_classifier(pooled_output)

        return request_logits, sub_request_logits
    #setp 3:  Training the Model
    from torch.utils.data import Dataset
from transformers import BertTokenizer

class EmailDataset(Dataset):
    def __init__(self, emails, request_labels, sub_request_labels, tokenizer):
        self.emails = emails
        self.request_labels = request_labels
        self.sub_request_labels = sub_request_labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.emails)

    def __getitem__(self, idx):
        email = self.emails[idx]
        request_label = self.request_labels[idx]
        sub_request_label = self.sub_request_labels[idx]

        encoding = self.tokenizer(email, padding='max_length', truncation=True, return_tensors='pt', max_length=128)

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'request_labels': torch.tensor(request_label, dtype=torch.long),
            'sub_request_labels': torch.tensor(sub_request_label, dtype=torch.long),
        }

# Example data
email_dataset = EmailDataset(emails, request_labels, sub_request_labels, tokenizer)

