# Theory Behind the Resume Parser

## Background
The old implementation of this project utilized a manually fine-tuned **spaCy 3** NER (Named Entity Recognition) model. This approach required a large, carefully annotated dataset to train the model to recognize specific resume fields (e.g., Name, Email, Skills, Experience, Education). If you wanted to extract a new type of field, you would have to re-annotate data and retrain the model from scratch.

## The Modern Approach: GLiNER
To modernize the project, we have transitioned from traditional fine-tuned spaCy models to **GLiNER (Generalist Model for Named Entity Recognition)**. 

### What is GLiNER?
GLiNER is a zero-shot NER model introduced in 2024. Unlike traditional models, GLiNER does not need to be fine-tuned on a specific set of labels. Instead, it accepts two inputs:
1. The text you want to parse.
2. A list of labels you want to extract (e.g., `["Name", "Phone", "Email", "Education", "Experience", "Skills"]`).

GLiNER uses bidirectional encoder representations (similar to BERT/DeBERTa) and contrastive learning to map the text segments and the label descriptions into a shared embedding space. If a text segment closely aligns with a label description, the model extracts it.

### Why is it Better?
- **Zero-Shot Capability**: You can add new labels dynamically without retraining.
- **Accuracy**: It leverages massive general-purpose pre-training, often outperforming small custom-trained models on specialized tasks.
- **Simplicity**: Removes the need for complex training pipelines and huge annotated datasets.

## Document Parsing
We use **PyMuPDF (fitz)** for PDF parsing and **python-docx** for DOCX parsing. These libraries represent the current state-of-the-art for fast, accurate text extraction in Python.
