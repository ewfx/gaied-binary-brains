import torch
from torch.nn.functional import softmax
import re
from preproocess import lemmatize_content
from llama3_together import classify_email

def clean_text(text: str, subject: str, attachments: str):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters

    if subject:
        subject = subject.lower()
        subject = re.sub(r"\s+", " ", subject)  # Remove extra spaces
        subject = re.sub(r"[^\w\s]", "", subject)  # Remove special characters

    if attachments:
        attachments = attachments.lower()
        attachments = re.sub(r"\s+", " ", attachments)  # Remove extra spaces
        attachments = re.sub(r"[^\w\s]", "", attachments)  # Remove special characters
    return text,subject,attachments

def predict_text(email_text: str, subject: str, attachments: str):
    email_text, subject,attachments  = clean_text(email_text, subject, attachments)

    email_text, subject,attachments  = lemmatize_content(email_text, subject,attachments)
    print("Email Text:", email_text)
    print("Subject:", subject)
    print("Attachments:", attachments)

    return classify_email(email_text, subject,attachments)
