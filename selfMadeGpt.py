
# TEXT EMBEDDING
import re
import json

class Tokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.special_tokens = ['[PAD]', '[UNK]', '[CLS]']

    def build_vocab(self, text):
        tokens = self._tokenize(text)
        vocab = sorted(set(tokens))
        vocab = self.special_tokens + vocab
        self.word_to_id = {word: i for i, word in enumerate(vocab)}
        self.id_to_word = {i: word for word, i in self.word_to_id.items()}

    def save_vocab(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.word_to_id, f)

    def load_vocab(self, filepath):
        with open(filepath, 'r') as f:
            self.word_to_id = json.load(f)
            self.id_to_word = {int(i): w for w, i in self.word_to_id.items()}

    def _tokenize(self, text):
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def encode(self, text):
        return [self.word_to_id.get(word, self.word_to_id['[UNK]']) for word in self._tokenize(text)]

    def decode(self, ids):
        return " ".join(self.id_to_word.get(i, '[UNK]') for i in ids)


# Usage example

text = "Hi, I am Frank. I like transformers!"
tokenizer = Tokenizer()
tokenizer.build_vocab(text)
tokenizer.save_vocab("vocab.json")  # Save vocab to disk

# Later or in another script
tokenizer2 = Tokenizer()
tokenizer2.load_vocab("vocab.json")

print("Vocab size:", len(tokenizer2.word_to_id))
print("Tokens:", tokenizer2.encode("I like transformers!"))
print("Decoded:", tokenizer2.decode(tokenizer2.encode("I like transformers!")))


# ROTARY POSITIONAL EMBEDDING (RoPE)

# PARALLELIZED MASKED MULTI-HEAD ATTENTION + FEED FORWARD

# RMSNORM or NORMFORMER

# GATED FEED FORWARD (with SwiGLU or GEGLU)

# DROPOUT + RESIDUAL CONNECTIONS

# FINAL RMSNORM / LAYER NORM

# TASK PREDICTION

# TASK CLASSIFIER