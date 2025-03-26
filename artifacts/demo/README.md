# 🚀 Project Name
GEN AI BASED Email Classification and OCR
## 📌 Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Challenges We Faced](#challenges-we-faced)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## 🎯 Introduction
Binary Brains Team: GENAI-EMAIL-TRAIAGE

AI email triage is an ​​advanced email management approach that uses AI algorithms to categorize and prioritize incoming emails. Unlike traditional email management methods that rely on manual sorting and filtering, AI email triage automates the process by learning from historical data and user behavior and training AI models to do the same.



## 🎥 Demo
🔗 [Live Demo](#) (if applicable)  
📹 [Video Demo](#) (if applicable)  
🖼️ Screenshots:
 ## attached test results code\test\BinaryBrain _TestResultsscreenshots.docx

![Screenshot 1](link-to-image)

## 💡 Inspiration
Proble statement:

The challenge of email overload is real. Professionals spend hours each day wading through cluttered inboxes, trying to prioritize important messages and respond promptly.
But imagine a world where your inbox is automatically organized, prioritized based on content, urgency, and relevance, and even responded to, freeing you up to focus on high-value work. That's the promise of AI email triage.

## ⚙️ What It Does
Binary Brains Team: GENAI-EMAIL-TRAIAGE


Acheived:
=> we have come up with an API wchi Accepts email in different formats (.pdf/.doc/.eml) along with attachments (.pdf/.doc) as  an input and returns "JSON" having classification details such as 
   -request type
   -sub request type
   -confidence score
   -urgency
   -keyPhrases
  details  as attributes

=> with the results we can determine 
   -improtance of email using "urgency" attribute 
   -classify email based on "request type" and "sub request type" 
   -can create "service reqeust" ticket based on "confidence score"

=>we can enhance requesttype and sub request type in future through configuration provided.

Limitations:
scope for scaling it to "service reqeuest generation tool" ingegration to create tickets.

## 🛠️ How We Built It

step 1: API will take the input file ( .docx/.pdf/.txt/.eml) ( email_Doc_triage.py)
step 2: Data preprocessing
         -Cleans and preprocesses email text for better model performance 
         -part of performing  data preprocessing we are doing cleaning of text ( predictor.py) and lemmetization(preproocess.py) using nltk
step 3: connect to LLAMA3  api using together API platform and perform email classification by
         -reading content from uploaded file (.eml/.pdf/.docx/.txt) 
         -along with attachment (.pdf/.docx/.txt) 



## 🚧 Challenges We Faced
started with BERT LLM for email classification  and spam detection.

->we were facing finetuning issues.
->one of us got space constraint issue as well as were connecting to model () 

*due to time constraints switched to Llama and able to come up with API*




## 🏃 How to Run
1. Clone the repository  
   ```sh
   git clone https://github.com/ewfx/gaied-binary-brains.git
   ```
2. Install dependencies  
   ```sh
      #Run below command to install all required libraries from project root folder
       pip install -r requirements.txt (for Python)
   ```
3. Run the project  
   ```sh
      # go to code/src folder and run below command
      # running Uvicorn with the module name where your FastAPI app is inside 
   uvicorn email_doc_triage:app --reload

   # Swagger 
   http://127.0.0.1:8000/docs
   ```

## 🏗️ Tech Stack


- 🔹 Frontend: API (Swagger docs, postman etc.,)
- 🔹 Backend: FastAPI, uvicorn
- 🔹 Other: LLAMA3 using together API platform

-Tools: Vscode
-Technology: Pythong programming language
-Framework: FASTAPI framework to connect to LLM(Llama)
-LLM used: Llama-3.3-70B-Instruct-Turbo
-platform to connect to LLM: https://api.together.xyz/

## 👥 Team
- **Yamini Pinapatruni** - [yamini-pinapatruni](#) ( yamini.pinapatruni@wellsfargo.com)
- **Divya Ambati** - [divzkala](#) (ambati.divyakala@wellsfargo.com)
