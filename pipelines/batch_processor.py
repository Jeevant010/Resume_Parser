import os
import json
import shutil
from pathlib import Path
from app.services.extractor import ResumeParserService
from pipelines.post_processing import aggregate_entities

def run_pipeline():
    print("Initializing Batch Processing Pipeline...")
    
    # Initialize the heavy ML model once
    parser = ResumeParserService()
    
    input_dir = Path("data/input")
    output_dir = Path("data/output")
    archive_dir = Path("data/archive")
    
    # Ensure directories exist
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in data/input/. Pipeline completed.")
        return
        
    print(f"Found {len(pdf_files)} resumes to process.")
    
    for pdf_path in pdf_files:
        filename = pdf_path.name
        print(f"Processing: {filename}")
        
        try:
            # 1. Extract raw text in chunks
            chunks = parser.extract_text(str(pdf_path))
            
            # 2. Run zero-shot ML extraction
            raw_entities = parser.parse_resume(chunks)
            
            # 3. Post-process and aggregate
            final_data = aggregate_entities(raw_entities)
            
            # 4. Save JSON output
            json_filename = f"{pdf_path.stem}.json"
            output_path = output_dir / json_filename
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_data, f, indent=4, ensure_ascii=False)
                
            # 5. Move PDF to archive
            archive_path = archive_dir / filename
            # Handle overwrite if file already exists in archive
            if archive_path.exists():
                archive_path.unlink()
            shutil.move(str(pdf_path), str(archive_path))
            
            print(f"  -> Successfully saved structured data to data/output/{json_filename}")
            
        except Exception as e:
            print(f"  -> ERROR processing {filename}: {e}")

if __name__ == "__main__":
    run_pipeline()
