import torch
from torch.nn.functional import softmax
import re
from preproocess import lemmatize
from llama3_together import classify_email

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    return text

def predict_text(email_text: str):
    email_text = clean_text(email_text)

    email_text = lemmatize(email_text)
    print("Email Text:", email_text)

    return classify_email(email_text)