import fitz
import os
from gliner import GLiNER

class ResumeParserService:
    def __init__(self):
        # Initialize the model on startup so we don't have to reload it per request
        print("Loading GLiNER model (Large v2.1)...")
        self.model = GLiNER.from_pretrained('urchade/gliner_large-v2.1')
        self.labels = ['Name', 'Email', 'Phone', 'Location', 'Experience', 'Education', 'Skills']
        print("GLiNER model loaded successfully.")
        
    def extract_text(self, file_path: str) -> list:
        """Extracts text from a PDF file in chunks (paragraphs) to improve NER accuracy."""
        chunks = []
        with fitz.open(file_path) as doc:
            for page in doc:
                blocks = page.get_text("blocks")
                for b in blocks:
                    block_text = b[4].strip()
                    if block_text:
                        chunks.append(block_text.replace('\n', ' '))
        return chunks
        
    def parse_resume(self, chunks: list) -> list:
        """Predicts entities dynamically using GLiNER on chunks."""
        all_entities = []
        for chunk in chunks:
            entities = self.model.predict_entities(chunk, self.labels)
            for e in entities:
                all_entities.append({"label": e["label"], "text": e["text"]})
        return all_entities
