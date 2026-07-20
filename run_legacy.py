import os
# Ensure the working directory is the project root
if os.path.basename(os.getcwd()) == 'R':
    os.chdir('..')
print("Working Directory:", os.getcwd())

!pip install -U spacy
!pip install PyMuPDF

import os
# Clone into R/ folder to keep it isolated
if not os.path.exists('R/CV-Parsing-using-Spacy-3'):
    os.system('git clone https://github.com/laxmimerit/CV-Parsing-using-Spacy-3.git R/CV-Parsing-using-Spacy-3')

import json
cv_data = json.load(open('R/CV-Parsing-using-Spacy-3/data/training/train_data.json', 'r'))
print("Loaded", len(cv_data), "training examples")

!python -m spacy init fill-config R/CV-Parsing-using-Spacy-3/data/training/base_config.cfg R/config.cfg

import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

def get_spacy_doc(file, data):
    nlp = spacy.blank("en")
    db = DocBin()
    for text, annot in tqdm(data):
        doc = nlp.make_doc(text)
        annot = annot['entities']
        ents = []
        entity_indices = []
        for start, end, label in annot:
            skip_entity = False
            for idx in range(start, end):
                if idx in entity_indices:
                    skip_entity = True
                    break
            if skip_entity:
                continue
            entity_indices = entity_indices + list(range(start, end))
            try:
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
            except:
                continue
            if span is None:
                file.write(str([start, end]) + "    " + str(text) + "\n")
            else:
                ents.append(span)
        try:
            doc.ents = ents
            db.add(doc)
        except:
            pass
    return db

from sklearn.model_selection import train_test_split
train, test = train_test_split(cv_data, test_size=0.3)

with open('R/error.txt', 'w', encoding='utf-8') as file:
    db_train = get_spacy_doc(file, train)
    db_train.to_disk('R/train_data.spacy')
    db_test = get_spacy_doc(file, test)
    db_test.to_disk('R/test_data.spacy')

!python -m spacy train R/config.cfg --output R/output --paths.train R/train_data.spacy --paths.dev R/test_data.spacy

import fitz
# If model didn't train successfully, uncomment below line to use pre-trained model:
# nlp = spacy.load('R/CV-Parsing-using-Spacy-3/nlp_model')
nlp = spacy.load('R/output/model-best')

pdf_path = 'data/Alice Clark CV.pdf'
text = " "
with fitz.open(pdf_path) as doc:
    for page in doc:
        text = text + str(page.get_text())

text = ' '.join(text.split())
doc = nlp(text)
for ent in doc.ents:
    print(f"{ent.label_.upper():<15}: {ent.text}")