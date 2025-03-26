from docx import Document
import PyPDF2
from docx import Document  # python-docx for DOCX files
from email import policy
from email.parser import BytesParser

# Function to read EML file and extract content along with attachments
def read_eml_file(file_path):
    with open(file_path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    subject = msg["subject"]
    email_body = ""
    attachments = []
    attachments_text = ""

    for part in msg.walk():
        content_type = part.get_content_type()
        content_disposition = part.get_content_disposition()

        if content_type == "text/plain":  # Extract email text
            email_body = part.get_payload(decode=True).decode(errors="ignore")

        elif content_disposition == "attachment":  # Extract attachments
            file_name = part.get_filename()
            file_bytes = part.get_payload(decode=True)
            attachments_text += "\n\n"+ bytes_to_string(file_bytes)
            attachments.append((file_name, file_bytes))

    return subject, email_body, attachments_text

def bytes_to_string(file_bytes, encoding="utf-8"):
    return file_bytes.decode(encoding, errors="ignore")


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

def extarct_text_from_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        return text

def get_file_contnet(file_path):
    content = ""
    subject = ""
    attachments_text = ""
    if file_path.endswith(".docx"):
        content = extract_text_from_docx(file_path)
    elif file_path.endswith(".pdf"):
        content = extract_text_from_pdf(file_path)
    elif file_path.endswith(".eml"):
        subject, content, attachments_text = read_eml_file(file_path)
    elif file_path.endswith(".txt"):
        content = extarct_text_from_text_file(file_path)
    return content, subject,attachments_text;


