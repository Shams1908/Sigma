from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
import uuid
import os
from pathlib import Path

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

analysis_store = {}

class UploadResponse(BaseModel):
    analysis_id: str
    filename: str
    file_size: int
    is_synthetic: bool
    status: str

@router.post("", response_model=UploadResponse)
async def upload_signal_file(
    file: UploadFile = File(...),
    is_synthetic: str = Form("false")
):
    analysis_id = str(uuid.uuid4())
    
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in ['iq', 'wav']:
        return {"error": "Invalid file type. Only .iq and .wav files are supported."}
    
    file_path = UPLOAD_DIR / f"{analysis_id}_{file.filename}"
    
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    is_synth = is_synthetic.lower() == "true"
    
    analysis_store[analysis_id] = {
        "filename": file.filename,
        "file_path": str(file_path),
        "file_size": len(content),
        "is_synthetic": is_synth,
        "status": "uploaded"
    }
    
    return UploadResponse(
        analysis_id=analysis_id,
        filename=file.filename,
        file_size=len(content),
        is_synthetic=is_synth,
        status="uploaded"
    )
