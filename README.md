# Byte Pair Encoding (BPE) Tokenizer

This project implements a **Byte Pair Encoding (BPE) Tokenizer**, widely used in NLP for efficient subword tokenization. The tokenizer can train from scratch using the English Wikipedia dataset untill a desired vocabulary size is reached or load pre-saved merge rules for real-time tokenization.

## Features
- **Trainable**: Learns merge rules to tokenize words into subword units.
- **Reusable**: Saves merge rules in JSON format for future use.
- **Interactive**: Tokenize user input in real-time.

## Dataset
The tokenizer uses the English Wikipedia dataset, available through the Hugging Face `datasets` library. During training:
- The first 15,000 documents of the dataset are processed.
- You can modify the number of documents in the script by changing the `num_docs` variable.

Ensure the dataset is available:
```python
from datasets import load_dataset
ds = load_dataset("lucadiliello/english_wikipedia", split="train")
```
## How to Use
1. **Train the Tokenizer**:
- If no merge rules are present, the script trains using the English Wikipedia dataset.
- Merge rules are saved to bpe_merge_rules.json.

2. **Load Pre-Saved Rules**:
- If bpe_merge_rules.json exists, the tokenizer will skip training and load the rules.

3. **Run the Script**:
  ```python
  python bpe_tokenizer.py
  ```
4. **Tokenize Strings**: Enter text when prompted to see the tokenized output.

## Example
```python
Enter a string to be tokenized (or 'q' to quit): 'Tokenization example'
Tokenized: ['to', 'k', 'en', 'iz', 'ation', 'exam', 'ple']
```
```python
Enter a string to be tokenized (or 'q' to quit): 'This paper introduces Random Multimodel Deep Learning'
Tokenized: ['this', 'p', 'ap', 'er', 'int', 'ro', 'du', 'ces', 'r', 'and', 'om', 'm', 'ul', 'tim', 'o', 'de', 'l', 'de', 'ep', 'le', 'ar', 'ning']
```
## Requirements
Install dependencies:
```python
pip install datasets
```



