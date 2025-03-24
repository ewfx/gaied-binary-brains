from transformers import BertForSequenceClassification, BertTokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertTokenizer

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
"Prinicipal+Interest":10
    # More Sub-Request Types...
}

emails = [
"I am writing to formally notify you regarding the closing of my account and to inquire about the reallocation fees associated with this process",
"I observed an inconsistency in the specific transaction details which I believe requires",
"I would like to initiate a transfer of funds between my accounts and would appreciate your assistance in processing this request",
"I recently noticed a fee labeled as Amendment Fee in my latest statement",
"After reviewing my financial situation, I would like to transfer a portion of the principal from my current loan/account to another account for better alignment with my financial goals",
"I am writing to request a decrease in the charges associated with my account and to discuss my commitment to maintaining a strong relationship with Wells Fargo",
"I am writing to request a change in my account commitment and discuss the possibility of an increase in my [e.g., credit limit, loan amount, deposit rate, etc.] due to my continued loyalty and growing financial needs",
"I am reaching out to inquire about the ongoing fees associated with my Wells Fargo account. After reviewing my recent statements, I noticed recurring fees that are being applied to my account",
"I am writing to inquire about the Letter of Credit Fee applied to my Wells Fargo account. I recently reviewed my statement and noticed a charge labeled as Letter of Credit Fee.",
"I would like to move the principal amount from an external source into my account."
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

