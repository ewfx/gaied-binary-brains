from transformers import BertTokenizer, BertForSequenceClassification,Trainer, TrainingArguments
import torch
import pytesseract
import pdfminer
from pdfminer.high_level import extract_text
from docx import Document
import torch

#from PIL import Image
import json
import os

# Initialize BERT model and tokenizer for text classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)  # E.g., 6 labels: spam, request type, sub request type
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to extract text from PDF attachments
def extract_pdf_text(pdf_path):
    text = extract_text(pdf_path)
    return text

# Function to extract text from DOCX attachments
def extract_docx_text(docx_path):
    doc = Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Function to extract text from image attachments using OCR
#def extract_text_from_image(image_path):
#    image = Image.open(image_path)
#    text = pytesseract.image_to_string(image)
#    return text

# Function to classify email content and attachment, then get confidence score
def classify_email_with_attachments(email_subject, email_body, attachment_paths=[]):
    # Combine email subject and body into one text input
    email_content = email_subject + "\n" + email_body

    # Extract text from attachments
    attachment_info = {}
    for attachment_path in attachment_paths:
        _, ext = os.path.splitext(attachment_path)
        if ext.lower() == ".pdf":
            attachment_info["pdf_attachment"] = {
                "filename": attachment_path,
                "page_count": len(extract_pdf_text(attachment_path).split('\n')) // 30,  # Estimate page count
                "extracted_text": extract_pdf_text(attachment_path)
            }
            email_content += "\n" + extract_pdf_text(attachment_path)
        elif ext.lower() == ".docx":
            attachment_info["docx_attachment"] = {
                "filename": attachment_path,
                "extracted_text": extract_docx_text(attachment_path)
            }
            email_content += "\n" + extract_docx_text(attachment_path)
        #elif ext.lower() in [".jpg", ".jpeg", ".png"]:
        #    attachment_info["image_attachment"] = {
        #        "filename": attachment_path,
        #        "image_text": extract_text_from_image(attachment_path)
        #    }
        #    email_content += "\n" + extract_text_from_image(attachment_path)
    
    # Tokenize the combined email content
    inputs = tokenizer(email_content, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Make predictions using the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Convert logits to probabilities using softmax
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Get predicted class and its confidence score
    predicted_class = torch.argmax(probs, dim=-1).item()
    confidence_score = probs[0][predicted_class].item()

    # Define request types and sub-request types as labels
    request_types = ['Adjustment','AU Transfer','Closing Notice','Commitment Change','Money Movement-inbound','Money Movement-outbound','Spam', 'Loan', 'Account', 'Fee Payment', 'Inquiry', 'Other']
    sub_request_types = ['Reallocation Fees', 'Amendement Fees','Reallocation Principal','Cashless Roll','Decrease','increase','Ongoing Fee',' Letter of Credit Fee','Principal','Interest','Prinicipal+Interest','Principal+Interest+Fee','Timebound','Foreign Currency','Spam', 'Payment Issue', 'Overpayment', 'Balance Inquiry', 'Account Query', 'Other']

    # Example of assigning labels
    request_type = request_types[predicted_class % len(request_types)]
    sub_request_type = sub_request_types[predicted_class % len(sub_request_types)]

    # Fake Spam score for demonstration
    spam_score = 0.05 if predicted_class != 0 else 0.95

    # Return the extended JSON result
    result = {
        'predicted_label': request_type,
        'confidence_score': confidence_score,
        'request_type': request_type,
        'sub_request_type': sub_request_type,
        'spam_score': spam_score,
        'attachment_info': attachment_info
    }

    return json.dumps(result, indent=4)

# Example email with attachments
email_subject = "Inquiry Regarding Ongoing Fee Payment"
#email_body = "Dear Support, I am facing an issue with my loan repayment due to delayed payment."
email_body = "I hope this email finds you well. My name is Divya, and I am customer of Wells Fargo with account number 12345. I am reaching out to seek clarification and assistance regarding an ongoing fee associated with my loan.Specifically, I would like to understand the following:The details of the ongoing fee, including its purpose and frequency.The payment schedule and due dates for this fee.Any options available to modify or waive this fee, if applicable.If there are any documents or additional information required from my end to facilitate this request, please let me know. I would appreciate it if you could provide a response at your earliest convenience.Thank you for your time and assistance. I look forward to your reply."
attachment_paths = ["./BinaryBrainTeam/artifacts/arch/LoanOverduepaymentProof.pdf"]

# Classify the email with attachments and get confidence score
#result = classify_email_with_attachments(email_subject, email_body, attachment_paths)
result = classify_email_with_attachments(email_subject, email_body)
print(result)