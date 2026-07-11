# Experimentations and Problem Logging

## 1. Initial State: The Legacy spaCy Pipeline

### Problem Definition
The project was originally ported from a Google Colab notebook utilizing `spaCy v3` for Custom Named Entity Recognition. The pipeline was designed to train a transformer-based NER model (`en_core_web_trf`) on a custom dataset of annotated resumes (in `.json` format, converted to `.spacy` binary format).

### Encountered Issues during Initial Experimentation
1. **Span Alignment Failures**: During the `get_spacy_doc` dataset preparation phase, we observed numerous errors (logged in `error.txt`). The `doc.char_span(start, end, label=label)` function continuously returned `None`.
   - **Root Cause**: The character offsets in the raw training JSON data (`[start, end]`) did not align perfectly with spaCy's strict tokenization boundaries (often due to trailing whitespaces or hidden PDF control characters like `\n` or `\t`).
   - **Math/Algorithmic Impact**: If the strict boundary alignment fails, the supervised learning process loses critical positive examples ($\mathcal{S}_{pos}$), starving the CRF layer of the necessary transitions.
2. **Environment Brittleness**: The notebook heavily relied on `Cupy` and specific CUDA setups for GPU acceleration (`!python -m spacy train ... --gpu-id 0`). This made local deployment highly unstable across different hardware (Windows, no GPU, mismatched CUDA versions).
3. **Catastrophic Forgetting and Label Rigidity**: Whenever we wanted to extract a new field (e.g., separating `Skills` into `Soft Skills` and `Hard Skills`), the output layer $W \in \mathbb{R}^{|\mathcal{T}| \times d_{model}}$ had to be resized, requiring a full retrain of the entire corpus.

## 2. Transition and Experimentation with Modern Approaches

To resolve these architectural limitations, we performed several design experiments.

### Experiment 1: Relaxing Span Alignment
- **Hypothesis**: Setting `alignment_mode="contract"` or `"expand"` in spaCy would salvage the misaligned training data.
- **Result**: While `error.txt` shrank, the model began hallucinating boundaries (e.g., extracting " Java" instead of "Java", or cutting off phone numbers). This polluted the embedding space during fine-tuning.

### Experiment 2: LLM (GPT-based) Prompting
- **Hypothesis**: Feed the extracted resume text into an LLM (like GPT-3.5 or a local LLaMA 3 8B) and ask for a JSON extraction.
- **Result**: Extremely high accuracy, but heavily bottlenecked by inference speed and context window token limits. For a batch of 1,000 resumes, this approach is computationally too expensive and slow for a fast FastAPI backend.

### Experiment 3: Zero-Shot Bi-Encoder (GLiNER)
- **Hypothesis**: Use a specialized, small-footprint contrastive model trained specifically for zero-shot NER (`GLiNER`).
- **Setup**: We integrated `gliner_medium-v2.1` (approx. ~350M parameters). 
- **Result**: 
  - **Flexibility**: We successfully extracted dynamic entities without any retraining.
  - **Speed**: Processing a resume takes $< 0.5$ seconds on CPU, easily outperforming LLMs.
  - **Robustness**: Extracted entities are structurally sound because the model evaluates span-to-label similarity in a high-dimensional continuous space, sidestepping the rigid token-alignment issues of spaCy.

## 3. Conclusion and Final Architecture
The experimentation phase concluded that maintaining an outdated supervised spaCy pipeline is mathematically and operationally inferior for resume parsing compared to modern Zero-Shot approaches.

The final architecture uses:
1. **Extraction**: `PyMuPDF` (for robust byte-level text layout extraction from PDFs) and `python-docx` for XML-based word documents.
2. **Inference Pipeline**: A stateless GLiNER predictor wrapped in a FastAPI service class, allowing for horizontal scaling without state bottlenecks.
