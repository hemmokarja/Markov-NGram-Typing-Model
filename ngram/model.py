import logging
import math
import multiprocessing
import pickle
import random
import time
from collections import defaultdict, Counter
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm


def _process_text_chunk_worker(args):
    text_chunk, chunk_idx, tokenizer, min_n, max_n, show_progress = args

    vocab = set()
    ngram_counts = {i: defaultdict(Counter) for i in range(min_n, max_n + 1)}
    context_counts = {i: Counter() for i in range(min_n, max_n + 1)}

    loop = tqdm(
        text_chunk,
        desc=f"Worker {chunk_idx}",
        disable=not show_progress,
        position=chunk_idx
    )
    for text in loop:
        tokens = tokenizer.tokenize(text)
        vocab.update(tokens)

        # generate g-grams for all sizes from min_n to max_n
        for gram_size in range(min_n, max_n + 1):
            for i in range(len(tokens) - gram_size + 1):
                context_end = i + gram_size - 1
                context = tuple(tokens[i:context_end])
                next_word = tokens[context_end]
                
                ngram_counts[gram_size][context][next_word] += 1
                context_counts[gram_size][context] += 1
    
    return vocab, ngram_counts, context_counts


class MarkovChainNGramModel:
    def __init__(self, tokenizer, n=2, min_n=None, smoothing=False, verbose=False):
        if n < 2:
            raise ValueError("n must be greater than 1")
        if min_n is not None and min_n < 2:
            raise ValueError("min_n must be greater than 1 if not None")

        self.tokenizer = tokenizer
        self.n = n
        self.min_n = min_n or n
        self.smoothing = smoothing
        self.vocab = None
        self.ngram_counts = None
        self.context_counts = None

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    def train_multiprocess(self, texts, num_processes, show_progress=False):
        start = time.time()

        num_processes = min(num_processes, multiprocessing.cpu_count())
        chunk_size = len(texts) // num_processes

        self.logger.info(
            f"Training with {num_processes} processes, chunk size: {chunk_size}"
        )

        text_chunks = [
            texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)
        ]

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            future_to_chunk = {}
            for chunk_idx, chunk in enumerate(text_chunks):
                args = (
                    chunk, chunk_idx, self.tokenizer, self.min_n, self.n, show_progress
                )
                future = executor.submit(_process_text_chunk_worker, args)
                future_to_chunk[future] = chunk_idx

            partial_results = []
            for future in futures.as_completed(future_to_chunk):
                result = future.result()
                partial_results.append(result)
                chunk_idx = future_to_chunk[future]
                self.logger.debug(f"Completed chunk {chunk_idx}")

        self.logger.info("Merging results...")
        vocab, ngram_counts, context_counts = self._merge_results(partial_results)

        self.vocab = vocab
        self.ngram_counts = ngram_counts
        self.context_counts = context_counts

        end = time.time()
        took = int(end - start)
        mins = took // 60
        secs = took % 60

        self.logger.info(f"Training completed in {mins:02d}:{secs:02d}")

    def train(self, texts, show_progress=False):
        start = time.time()

        self.logger.info("Training with single process")

        vocab = set()
        ngram_counts = {i: defaultdict(Counter) for i in range(self.min_n, self.n + 1)}
        context_counts = {i: Counter() for i in range(self.min_n, self.n + 1)}

        for text in tqdm(texts, disable=not show_progress):
            tokens = self.tokenizer.tokenize(text)
            vocab.update(tokens)

            for gram_size in range(self.min_n, self.n + 1):
                for i in range(len(tokens) - gram_size + 1):
                    context_end = i + gram_size - 1
                    context = tuple(tokens[i:context_end])
                    next_word = tokens[context_end]

                    ngram_counts[gram_size][context][next_word] += 1
                    context_counts[gram_size][context] += 1

        self.vocab = vocab
        self.ngram_counts = ngram_counts 
        self.context_counts = context_counts

        end = time.time()
        took = int(end - start)
        mins = took // 60
        secs = took % 60

        self.logger.info(f"Training completed in {mins:02d}:{secs:02d}")

    def _merge_results(self, partial_results):
        final_ngram_counts = {
            i: defaultdict(Counter) for i in range(self.min_n, self.n + 1)
        }
        final_context_counts = {i: Counter() for i in range(self.min_n, self.n + 1)}
        final_vocab = set()

        for vocab, ngram_counts, context_counts in partial_results:
            final_vocab.update(vocab)

            for gram_size in range(self.min_n, self.n + 1):
                for context, word_counts in ngram_counts[gram_size].items():
                    final_ngram_counts[gram_size][context].update(word_counts)
                final_context_counts[gram_size].update(context_counts[gram_size])

        return final_vocab, final_ngram_counts, final_context_counts

    def _get_next_word_probs(self, context, current_n=None):
        """Get probability distribution for next word given context."""
        if current_n is None:
            current_n = self.n

        context = tuple(context[-(current_n - 1):])

        if (context in self.ngram_counts[current_n] and 
            self.context_counts[current_n][context] > 0):
            
            next_words = self.ngram_counts[current_n][context]
            total = self.context_counts[current_n][context]
            vocab_size = len(self.vocab)

            if self.smoothing:
                probs = {}
                for word in self.vocab:
                    count = next_words.get(word, 0)
                    probs[word] = (count + 1) / (total + vocab_size)
                return probs
            else:
                return {word: count / total for word, count in next_words.items()}
        
        # Backoff to shorter context
        elif current_n > self.min_n:
            return self.get_next_word_probs(context, current_n - 1)

        # Uniform distribution fallback
        return {word: 1 / len(self.vocab) for word in self.vocab}

    def generate_text(self, text, max_length=50, temperature=1.0):
        """Generate text using the trained model."""
        context = self.tokenizer.tokenize(text)

        if len(context) <= self.min_n - 1:
            self.logger.warning(
                f"Seed context only {len(context)} tokens, expected at least "
                f"{self.min_n - 1}."
            )

        generated = []
        
        for _ in range(max_length):
            probs = self._get_next_word_probs(context)

            if temperature != 1.0:
                for word in probs:
                    probs[word] = math.pow(probs[word], 1.0 / temperature)
                total = sum(probs.values())
                probs = {word: prob / total for word, prob in probs.items()}

            words, weights = zip(*probs.items())
            next_word = random.choices(words, weights=weights)[0]

            generated.append(next_word)
            context = context[1:] + [next_word]

        return self.tokenizer.decode(generated)

    def save(self, filepath):
        self.logger.info("Saving model state...")
        state = {
            "vocab": self.vocab,
            "ngram_counts": self.ngram_counts,
            "context_counts": self.context_counts,
            "n": self.n,
            "min_n": self.min_n,
            "smoothing": self.smoothing
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        
        self.logger.info(f"Model state saved to '{filepath}'")

    def load(self, filepath):
        self.logger.info("Loading model state...")
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.vocab = data["vocab"]
        self.ngram_counts = data["ngram_counts"]
        self.context_counts = data["context_counts"]
        self.n = data["n"]
        self.min_n = data["min_n"]
        self.smoothing = data["smoothing"]

        self.logger.info(f"Model state loaded.")
