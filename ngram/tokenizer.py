import os
import re

from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NaiveTokenizer:
    def __init__(self):
        pass

    def tokenize(self, text):
        # Pattern explanation:
        # \w+'\w+ : matches contractions like "I'm", "it's", "don't"
        # \w+ : matches regular words
        # [.!?] : matches sentence-ending punctuation
        return re.findall(r"\w+'\w+|\w+|[.!?]", text.lower())

    def decode(self, tokens):
        return " ".join(tokens)


class HugginfaceTokenizer:
    def __init__(self, tokenizer_name="gpt2"):
        
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def tokenize(self, text):
        tokens = []
        chunk_size = self.tokenizer.model_max_length
        for i in range(0, len(text), chunk_size):
            chunk = text[i: i + chunk_size]
            tokens += self.tokenizer.tokenize(chunk)
        return tokens

    def decode(self, tokens):
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return self.tokenizer.decode(input_ids, clean_up_tokenization_spaces=True)
