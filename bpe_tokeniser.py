import json
import re
import os
from collections import Counter
from datasets import load_dataset

'''This code implements a Byte Pair Encoding (BPE) tokenizer python class that learns merge rules for efficiently breaking down words into subword 
units. If merge rules(in json format) is already present in the current directory, it will load and use it for tokenization. Otherwise, 
it will train the tokenizer from scratch untill desired vocabulory size is reached using the the English Wikipedia dataset to generate the merge rules.'''

class BPE_Tokenizer:
    def __init__(self, vocab_size=700, initial_vocab=None):
        self.vocab_size = vocab_size
        self.vocab = set(initial_vocab or [])
        self.token_stats = {
            'frequencies': Counter(),
            'splits': {},
            'merge_rules': {}
        }

    def preprocess_corpus(self, corpus):
        for doc in corpus:
            words = re.findall(r'\b[a-z]+\b', doc.lower())
            for word in words:
                self.token_stats['frequencies'][word] += 1
                if word not in self.token_stats['splits']:
                    char_split = list(word)
                    self.token_stats['splits'][word] = char_split
                    self.vocab.update(char_split)

    def extract_pair_frequencies(self):
        pair_counts = Counter()
        for word, freq in self.token_stats['frequencies'].items():
            splits = self.token_stats['splits'][word]
            if len(splits) > 1:
                for i in range(len(splits) - 1):
                    pair = (splits[i], splits[i + 1])
                    pair_counts[pair] += freq
        return pair_counts

    def apply_merge(self, most_common_pair):
        a, b = most_common_pair
        merged_token = a + b

        for word in self.token_stats['splits']:
            new_split = []
            i = 0
            while i < len(self.token_stats['splits'][word]):
                if (i < len(self.token_stats['splits'][word]) - 1 and
                        self.token_stats['splits'][word][i] == a and
                        self.token_stats['splits'][word][i + 1] == b):
                    new_split.append(merged_token)
                    i += 2
                else:
                    new_split.append(self.token_stats['splits'][word][i])
                    i += 1

            self.token_stats['splits'][word] = new_split

        self.vocab.add(merged_token)
        self.token_stats['merge_rules'][(a, b)] = merged_token

    def train(self, corpus):
        self.preprocess_corpus(corpus)

        while len(self.vocab) < self.vocab_size:
            pair_frequencies = self.extract_pair_frequencies()

            if not pair_frequencies:
                break

            most_common_pair = pair_frequencies.most_common(1)[0][0]
            self.apply_merge(most_common_pair)

            progress = (len(self.vocab) / self.vocab_size) * 100
            print(f"\rTraining Progress: {progress:.2f}%", end="")

        print("\nTraining complete.")
        return self.token_stats['merge_rules']

    def save_merge_rules(self, filename):
        serializable_rules = {
            f"{k[0]}_{k[1]}": v for k, v in self.token_stats['merge_rules'].items()
        }
        with open(filename, 'w') as f:
            json.dump(serializable_rules, f)

    @classmethod
    def load_merge_rules(cls, filename):
        with open(filename, 'r') as f:
            loaded_rules = json.load(f)
        merge_rules = {
            tuple(k.split('_')): v for k, v in loaded_rules.items()
        }
        return merge_rules

    def tokenize(self, text, merge_rules):
        words = re.findall(r'\b[a-z]+\b', text.lower())
        tokens = [list(word) for word in words]

        for (a, b), merged in merge_rules.items():
            for doc_idx, doc_tokens in enumerate(tokens):
                new_tokens = []
                i = 0
                while i < len(doc_tokens):
                    if (i < len(doc_tokens) - 1 and
                            doc_tokens[i] == a and
                            doc_tokens[i + 1] == b):
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(doc_tokens[i])
                        i += 1
                tokens[doc_idx] = new_tokens

        return [token for sublist in tokens for token in sublist]


if __name__ == "__main__":
    merge_rules_file = 'bpe_merge_rules.json'

    if os.path.exists(merge_rules_file):
        print("Loading merge rules from file...")
        merge_rules = BPE_Tokenizer.load_merge_rules(merge_rules_file)
    else:
        print("Merge rules file not found. Starting training...")
        num_docs = 15000
        ds = load_dataset("lucadiliello/english_wikipedia", split="train")
        corpus = [doc['maintext'] for doc in ds.select(range(num_docs))]

        tokenizer = BPE_Tokenizer(vocab_size=700, initial_vocab=list('abcdefghijklmnopqrstuvwxyz'))
        merge_rules = tokenizer.train(corpus)
        tokenizer.save_merge_rules(merge_rules_file)

    while True:
        inp = input("Enter a string to be tokenized (or 'q' to quit): ")
        if inp.lower() == 'q':
            break
        tokenizer = BPE_Tokenizer()
        print("Tokenized:", tokenizer.tokenize(inp, merge_rules))
