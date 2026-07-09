!pip install spacy-transformers
!pip install -U spacy

# --- CELL ---

import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import json

# --- CELL ---

spacy.__version__

# --- CELL ---

!nvidia-smi

# --- CELL ---

!git clone https://github.com/laxmimerit/CV-Parsing-using-Spacy-3.git

# --- CELL ---

cv_data = json.load(open("C:\\order\\Desktop\\Resume_Parser\\R\\CV-Parsing-using-Spacy-3\\data\\training\\train_data.json", 'r'))

# --- CELL ---

len(cv_data)

# --- CELL ---

!python -m spacy init fill-config C:\order\Desktop\Resume_Parser\R\CV-Parsing-using-Spacy-3\data\training\base_config.cfg

# --- CELL ---

cv_data

# --- CELL ---

# cv_data = trim_entry_spans(cv_data)

# --- CELL ---

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
      if skip_entity == True:
        continue

      entity_indices = entity_indices + list(range(start, end))

      try :
        span = doc.char_span(start, end, label=label, alignment_mode="strict")
      except:
        continue

      if span is None:
        err_data = str([start, end]) + "    " + str(text) + "\n"
        file.write(err_data)

      else:
        ents.append(span)

    try:
      doc.ents = ents
      db.add(doc)
    except:
      pass

  return db




# --- CELL ---

from sklearn.model_selection import train_test_split
train, test = train_test_split(cv_data, test_size=0.3)

# --- CELL ---

len(train), len(test)

# --- CELL ---

file = open('error.txt', 'w')
db = get_spacy_doc(file, train)
db.to_disk('train_data.spacy')

db = get_spacy_doc(file, test)
db.to_disk('test_data.spacy')

file.close

# --- CELL ---

len(db.tokens)

# --- CELL ---

!pip install Cupy

# --- CELL ---

!python -m spacy train C:\order\Desktop\Resume_Parser\R\CV-Parsing-using-Spacy-3\data\training\base_config.cfg --output ./output --paths.train ./train_data.spacy --paths.dev ./test_data.spacy --gpu-id 0

# --- CELL ---

nlp = spacy.load("/content/output/model-best")

# --- CELL ---

doc = nlp("My name is John Doe. I am a Software Engineer. i have a experience of 5 years")
for ent in doc.ents:
  print(ent.text, '>>>' ,ent.label_)

# --- CELL ---

!pip install PyMuPDF

# --- CELL ---

import sys, fitz

# --- CELL ---

fname = '/content/CV-Parsing-using-Spacy-3/data/test/Alice Clark CV.pdf'
doc = fitz.open(fname)

# --- CELL ---

text = " "
for page in doc:
  text = text + str(page.get_text())

# --- CELL ---

text = text.strip()

# --- CELL ---

text = ' '.join(text.split())

# --- CELL ---

text

# --- CELL ---

doc = nlp(text)
for ent in doc.ents:
  print(ent.text, '>>>' ,ent.label_)