# 🚀 Project Name

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

![Screenshot 1](link-to-image)

## 💡 Inspiration
Proble statement:

The challenge of email overload is real. Professionals spend hours each day wading through cluttered inboxes, trying to prioritize important messages and respond promptly.
But imagine a world where your inbox is automatically organized, prioritized based on content, urgency, and relevance, and even responded to, freeing you up to focus on high-value work. That's the promise of AI email triage.

## ⚙️ What It Does
Binary Brains Team: GENAI-EMAIL-TRAIAGE


Acheived:
=> we have come up with an API wchi Accepts email in different formats (.pdf/.doc/.eml) along with attachments (.pdf/.doc) as  an input and returns "JSON" having classification details such as request type/ sub request type/ confidence score/ urgency/ key word identification details information as parameters 

## 🛠️ How We Built It




## 🚧 Challenges We Faced
started with BERT LLM for email classification  and spam detection.

->we were facing finetuning issues.
->one of us got space constraint as were connecting to model () space issues)

*due to time constraints switched to Llama and able to come up with API alone*
scope for scaling it to service management tool ingegration to create tickets.


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