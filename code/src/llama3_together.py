import requests
import json
import re
from load_config import categories

API_URL = "https://api.together.xyz/v1/completions"
HEADERS = {"Authorization": "Bearer 9dc633e600ea616eb32f0b7b786bd0e6cee3a8e390e6e199b0ef5bf1c1b042d8",
           "Content-Type": "application/json"}

# Function to determine urgency level based on keywords
def determine_importance(email_text: str) -> str:
    urgent_keywords = {"urgent", "immediate", "asap", "important", "critical"}
    if any(word in email_text.lower() for word in urgent_keywords):
        return "High"
    return "Normal"

# Function to classify email request type
def classify_email(email_text: str, subject: str, attachments: str):
    # Define a structured prompt with a format request
    importance = determine_importance(email_text)

    prompt =  f"""
    You are an AI assistant specializing in banking email classification.

    ### Task:
    Analyze the email and classify it into a category and subcategory. Additionally, extract important key phrases and determine the urgency level.

     ### Email:
    "{email_text}"

     ### Email Subject: (optional)
    "{subject}"

     ### Email attachments: (optional)
    "{attachments}"

    ### Email Details:
    - **Urgency Level**: {importance}

    ### Categories & Subcategories:
    {categories}

     ### Response Format:
    Strictly return JSON in the format below, without any extra text:
    {{
    "category": "<category>",
    "subcategory": "<subcategory>",
    "key_phrases": "<key_phrases>",
    "confidence": <confidence_score>,
    "urgency": "<importance>"
    }}

    ### Rules:
    1. Return **only** the JSON response—no explanations, extra text, or preamble.
    2. Ensure `"confidence"` is a float value between **0 and 1**.
    3. Do **not** add extra keys, values, or responses outside JSON.
    """
    
    payload = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "prompt": prompt,
        "temperature": 0.2,  # Lower for deterministic output
        "top_p": 0.9,
        "max_tokens": 100
    }

    result = requests.post(API_URL, headers=HEADERS, json=payload)
    

    try:
        response = result.json()
        print(response)
        response_text = response["choices"][0]["text"].strip()
        response_text = re.sub(r'`+', ' ', response_text)
        
        # Attempt to parse JSON from the response
        structured_result = json.loads(response_text)
        print("Response", structured_result)
        return structured_result

    except json.JSONDecodeError:
         print("Error: Model did not return a valid JSON response.")
         print("Raw Response:", response_text)
         print(response_text)
         return 
    
    
