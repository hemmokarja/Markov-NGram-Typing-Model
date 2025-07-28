import logging

import datasets

from ngram.model import MarkovChainNGramModel
from ngram.tokenizer import HugginfaceTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

N = 5
MIN_N = 3
NUM_PROCESSES = 10

NUM_TEXTS = 100_000


def _get_texts(num_texts=NUM_TEXTS):
    """
    Load a specified number of short stories from the TinyStories dataset.

    Each entry is a short story in natural English, making it a perfect fit for
    demonstrating n-gram language modeling.
    """
    dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")
    return dataset["text"][:num_texts]


def main():
    tokenizer = HugginfaceTokenizer("gpt2")
    model = MarkovChainNGramModel(tokenizer, n=N, min_n=MIN_N)

    texts = _get_texts()

    model.train_multiprocess(texts, NUM_PROCESSES, show_progress=True)

    model.save("state.pt")

if __name__ == "__main__":
    main()
