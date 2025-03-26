from docx import Document
import PyPDF2
import email
import os

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_eml(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        msg = email.message_from_file(f)
        return msg.get_payload()

def extarct_text_from_text_file(file_path):
    # Read content and perform prediction
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        return text

def get_file_contnet(file_path):
    if file_path.endswith(".docx"):
        content = extract_text_from_docx(file_path)
    elif file_path.endswith(".pdf"):
        content = extract_text_from_pdf(file_path)
    elif file_path.endswith(".eml"):
        content = extract_text_from_eml(file_path)
    elif file_path.endswith(".txt"):
        content = extarct_text_from_text_file(file_path)
    return content;


