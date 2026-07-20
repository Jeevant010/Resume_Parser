from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
from app.services.extractor import ResumeParserService
from app.models.schemas import ParseResponse

router = APIRouter()

# Instantiate the service once so the GLiNER model remains loaded in memory
parser_service = ResumeParserService()

@router.post("/parse", response_model=ParseResponse, summary="Parse Resume PDF")
async def parse_resume(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    temp_path = f"data/temp_{file.filename}"
    try:
        # Save uploaded file to temp path
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. Extract raw text
        text = parser_service.extract_text(temp_path)
        
        # 2. Parse text with ML model
        entities = parser_service.parse_resume(text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return ParseResponse(filename=file.filename, entities=entities)
