import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLP resources
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Load banking-specific keywords from a file
def load_keywords(file_path):
    with open(file_path, "r") as file:
        keywords = {line.strip().lower() for line in file.readlines()}
    return keywords

# Define domain-specific keywords for banking loan services
domain_keywords = load_keywords("../../config/banking_keywords.txt")


# Load stopwords
stop_words = set(stopwords.words('english'))

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize_content(text: str, subject: str, attachments: str):
    text = lemmatize(text)
    if subject:
        subject = lemmatize(subject)
    if attachments:
        attachments = lemmatize(attachments)
    return text, subject, attachments

def lemmatize(text):
    """Cleans and preprocesses email text for better model performance."""
    print("Original Text:", text)
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize words
    words = word_tokenize(text)
    
    # Remove stopwords and apply lemmatization
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # Keep only domain-specific keywords (optional)
    #cleaned_words = [word for word in cleaned_words if word in domain_keywords]

    cleaned_text = " ".join(cleaned_words)

    
    print("Cleaned Text:", cleaned_text)
    
    return " ".join(cleaned_words)



