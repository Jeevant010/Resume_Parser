# Mathematical Foundations of Named Entity Recognition

## 1. Introduction to the Problem Space
Resume parsing represents a complex Information Extraction (IE) problem. A resume is semi-structured or unstructured text. The goal is to accurately map a sequence of tokens $X = (x_1, x_2, \dots, x_n)$ to a sequence of tags $Y = (y_1, y_2, \dots, y_n)$ where $y_i \in \mathcal{T}$ (the set of entity tags like `Name`, `Email`, `Skills`).

The primary difficulty lies in the variability of formatting and phrasing across different resumes, making rigid regular expressions brittle and ineffective.

## 2. Evolution of NER Architectures

### 2.1 Conditional Random Fields (CRFs)
Historically, NER was solved using Linear-Chain CRFs. A CRF models the conditional probability $P(Y|X)$ directly.
Given a sequence of words $X$, the probability of a label sequence $Y$ is defined as:
$$ P(Y|X) = \frac{1}{Z(X)} \exp\left( \sum_{i=1}^n \sum_{j=1}^k \lambda_j f_j(y_{i-1}, y_i, X, i) \right) $$
Where:
- $f_j$ is a feature function (e.g., "is the word capitalized?", "is it followed by a number?").
- $\lambda_j$ is the learned weight for the feature function.
- $Z(X)$ is the normalization factor (partition function) computed using the Forward-Backward algorithm:
$$ Z(X) = \sum_{Y'} \exp\left( \sum_{i=1}^n \sum_{j=1}^k \lambda_j f_j(y'_{i-1}, y'_i, X, i) \right) $$

**Drawbacks in Resume Parsing:** CRFs rely heavily on manual feature engineering and struggle with long-range dependencies (e.g., linking a degree mentioned at the end of a resume to a name at the top).

### 2.2 BiLSTM-CRF Architecture
To capture long-range dependencies, Bidirectional Long Short-Term Memory (BiLSTM) networks were introduced.
For a given token $x_i$, its word embedding is $e_i \in \mathbb{R}^d$. The BiLSTM computes forward and backward hidden states:
$$ \overrightarrow{h_i} = \text{LSTM}_{fwd}(e_i, \overrightarrow{h_{i-1}}) $$
$$ \overleftarrow{h_i} = \text{LSTM}_{bwd}(e_i, \overleftarrow{h_{i+1}}) $$
The concatenated state $h_i = [\overrightarrow{h_i}; \overleftarrow{h_i}]$ represents the contextualized word representation. This $h_i$ is then fed into a CRF layer instead of manual features, allowing the model to learn both context and sequence constraints (e.g., an `I-Skill` must follow a `B-Skill`).

## 3. The Transformer Era (BERT & RoBERTa)
Transformers abandoned recurrence entirely in favor of the **Self-Attention Mechanism**.

### 3.1 Scaled Dot-Product Attention
For a matrix of queries $Q$, keys $K$, and values $V$ (all projected from the input embeddings):
$$ \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V $$
Where $d_k$ is the dimension of the keys. This allows the model to compute a weighted sum of all words in a sentence simultaneously, completely solving the long-range dependency problem.

### 3.2 Fine-Tuned Token Classification
In models like spaCy's transformer pipeline (or BERT-NER), a dense layer is placed on top of the final transformer hidden state $h_i$ for each token:
$$ P(y_i | X) = \text{softmax}(W \cdot h_i + b) $$
Where $W \in \mathbb{R}^{|\mathcal{T}| \times d_{model}}$.

**The Problem:** If you train this model for $N$ labels (e.g., `["Name", "Skill"]`), the matrix $W$ is fixed. If you want to add a new label `["Certifications"]`, you must re-initialize $W$ and fine-tune the entire model again on thousands of examples.

## 4. Modern Approach: GLiNER (Generalist NER)
GLiNER solves the fixed-label problem by using **Contrastive Learning** and a **Bi-Encoder Architecture**. It maps both the *text spans* and the *label descriptions* into a shared embedding space.

### 4.1 Span Representations
Instead of token-level classification, GLiNER uses span-level representations.
Let a span $s_{ij}$ be the sequence of tokens from index $i$ to $j$. Its representation $v_{ij}$ is computed by combining the boundary token representations:
$$ v_{ij} = [h_i ; h_j ; \text{MLP}_\text{width}(j - i)] $$
Where $\text{MLP}_\text{width}$ embeds the length of the span.

### 4.2 Label Representations
Let $L = \{l_1, l_2, \dots, l_K\}$ be the list of target entities (e.g., "Software skill", "University name").
A text encoder (like a BERT encoder) computes a representation for each label text:
$$ c_k = \text{Encoder}(l_k) $$

### 4.3 Similarity and Extraction
The probability that a text span $s_{ij}$ belongs to the entity class $l_k$ is given by the sigmoid of the dot product (or cosine similarity) of their representations:
$$ P(l_k | s_{ij}) = \sigma(v_{ij} \cdot c_k) $$

### 4.4 Contrastive Loss Function during Pre-training
During GLiNER's massive pre-training on diverse datasets, it optimizes a contrastive loss that pushes the embeddings of matching spans and labels closer together, and non-matching ones apart:
$$ \mathcal{L} = -\sum_{s \in \mathcal{S}_{pos}} \log \sigma(v_s \cdot c_{true}) - \sum_{s \in \mathcal{S}_{neg}} \log(1 - \sigma(v_s \cdot c_{false})) $$

**Why this matters for your pipeline:**
Because the labels are mathematically treated as *inputs* rather than *fixed weights*, you can query the model with *any* label string at inference time. This zero-shot capability makes it the mathematically optimal choice for parsing highly variable documents like resumes without retraining.
