# 🔁 Markov Chain N-Gram Text Generator

A simple and educational implementation of a Markov Chain-based N-Gram language model for text generation. This project demonstrates core natural language processing concepts such as tokenization, probabilistic modeling, and backoff strategies in N-Gram language models.

## 📚 About the Model

This model builds a **probabilistic language generator** using the **Markov property**, where the probability of a word depends only on a fixed number (N−1) of preceding words (context).

### 🔢 How It Works

- **N-Grams**: For a given N (e.g., 5), the model learns frequencies of token sequences of length N and uses them to generate the next likely token.
- **Markov Chain**: Treats token prediction as a state transition — the next state (word) depends only on the current context (N−1 previous tokens).
- **Backoff Strategy**: If a particular N-gram context is unseen, the model recursively backs off to (N−1)-grams, continuing until a known context is found or ultimately falling back to a uniform distribution.
- **Tokenization**: Uses Hugging Face's tokenizer ecosystem to perform flexible and robust subword tokenization.
- **Smoothing**: (Optional) Adds Laplace smoothing to handle sparsity by assigning nonzero probabilities to unseen words.

### ⚠️ Limitations

This model is intended for educational purposes and lacks the depth, generalization, and contextual understanding of modern neural language models (e.g., transformers). It performs best on small, well-structured datasets and is not suitable for tasks requiring nuanced comprehension, long-range dependencies, or semantic consistency.

## 🚀 Features

- ✅ Markov Chain-based probabilistic text generation
- ✅ Configurable N-Gram size with fallback to lower-order models
- ✅ Multi-process training support for large datasets
- ✅ Hugging Face tokenizer integration
- ✅ Save/load model state

## 🛠️ Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for package and environment management.

```bash
uv sync
```

## 🧠 Training the Model

Run the following to train and save the model:

```bash
uv run main.py
```

You can toggle between:
- **Single-process training**: `model.train(...)`
- **Multiprocess training**: `model.train_multiprocess(..., num_processes=...)`

### 📚 Dataset

The training script loads text data from the [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.

> **TinyStories** is a collection of short, synthetically generated stories using GPT-based models. The dataset is clean, small in vocabulary, and ideal for experiments in language modeling and generative tasks involving basic syntax and semantics.

## 🧪 Generating Text

Explore the usage and text generation capabilities with the provided example notebook:

**example.ipynb**

## 🧾 Example

```python
from transformers import AutoTokenizer
from markov_ngram import MarkovChainNGramModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = MarkovChainNGramModel(tokenizer=tokenizer, n=5, smoothing=True)

model.train_multiprocess(texts=my_corpus, num_processes=4)

model.generate_text("The future of AI", max_length=50)
```

## 📘 Theory

The **Markov Chain N-Gram model** approximates the probability of the next word `wₙ` given a history of `N−1` words `(wₙ₋ₙ₊₁, ..., wₙ₋₁)` as:

`P(wₙ | wₙ₋ₙ₊₁, ..., wₙ₋₁) = Count(wₙ₋ₙ₊₁, ..., wₙ) / Count(wₙ₋ₙ₊₁, ..., wₙ₋₁)`

When unseen sequences are encountered, the model **backs off** to shorter contexts to maintain robustness.

## 📝 License

This project is licensed under the MIT License.
