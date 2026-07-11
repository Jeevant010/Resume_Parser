# Working of the Resume Parser

## Overview
This document explains the architecture and operational flow of the modern Resume Parser application.

## Directory Structure
- **`R/`**: Contains only Jupyter Notebooks for experimentation, prototyping, and testing models independently of the main application.
- **`app/`**: The core FastAPI application directory. It contains all production code.
- **`manuals/`**: Contains theory and functional documentation (like this file).
- **`pipelines/`**: Stores scripts for testing, data processing, and CI/CD operations.
- **`data/`**: A storage directory for sample resumes and output logs.

## Application Flow
1. **File Upload**: The user uploads a resume (PDF, DOCX, or TXT) via the FastAPI endpoint (`/parse`).
2. **Text Extraction**: The `services/extractor.py` module identifies the file type and uses the appropriate library (`PyMuPDF` for PDF, `python-docx` for DOCX) to extract raw text.
3. **Entity Recognition**: The raw text is passed to `services/parser.py`. This service initializes the `GLiNER` model (downloading the weights on first run) and asks it to identify specific entities like `["Name", "Email", "Phone", "Skills", "Experience", "Education"]`.
4. **Formatting**: The extracted entities are structured into a JSON response defined by Pydantic models in `app/models/schemas.py`.
5. **Response**: The API returns the structured JSON to the client.

## How to Run the App
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```
3. Open `http://localhost:8000/docs` in your browser to interact with the API using Swagger UI.
