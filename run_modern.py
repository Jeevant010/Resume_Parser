import os
# Ensure the working directory is the project root
if os.path.basename(os.getcwd()) == 'R':
    os.chdir('..')
print("Working Directory:", os.getcwd())

!pip install gliner pymupdf python-docx

import fitz  # PyMuPDF
import os

# Path is relative to project root
pdf_path = 'data/Alice Clark CV.pdf'
text = ''
if os.path.exists(pdf_path):
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text() + ' '
    text = ' '.join(text.split())
    print('Extracted Text length:', len(text))
else:
    print('Sample PDF not found at', pdf_path)


from gliner import GLiNER

if text:
    model = GLiNER.from_pretrained('urchade/gliner_medium-v2.1')
    labels = ['Name', 'Email', 'Phone', 'Location', 'Experience', 'Education', 'Skills']
    entities = model.predict_entities(text, labels)
    for entity in entities:
        print(f"{entity['label']}: {entity['text']}")
