from fastapi import FastAPI, UploadFile, File
import shutil
from services.pdf_service import extract_text_from_pdf
from services.rag_service import process_text

app = FastAPI()

@app.get("/")
def home():
    return {"message": "NexusAI Backend Running 🚀"}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):

    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    extracted_text = extract_text_from_pdf(file_path)
    
    # Process text into chunks + embeddings
    chunks = process_text(extracted_text)


    return {
        "filename": file.filename,
        "chunk_created": len(chunks),
        "preview_chunk": chunks[0]
    }