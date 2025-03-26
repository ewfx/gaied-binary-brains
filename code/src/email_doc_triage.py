from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os
from predictor import predict_text
from fileread import get_file_contnet


app = FastAPI()

UPLOAD_DIR = "../../uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def home():
    return {"message": "Hello, Welcome to the email document triage application! You can upload an email or document to determine the type of request for triaging"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save file to disk for future training
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    print(file_path)
    
    text, subject, attachments = get_file_contnet(file_path)
    result = predict_text(text,subject, attachments)

    print("result:", result)
    
    return JSONResponse(content={
        "filename": file.filename,
        "content_type": file.content_type,
        "message": "File uploaded successfully",
        "request type": result["category"],
        "sub request type": result["subcategory"],
        "keyPhrases": result["key_phrases"],
        "confidence": result["confidence"],
        "urgency": result["urgency"]
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
