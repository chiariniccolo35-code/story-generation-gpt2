# Custom BPE Tokenizer

This folder contains the trained Byte-Pair Encoding (BPE) tokenizer for the story generation task.

## Files

- **tokenizer.json** — The trained BPE tokenizer weights and vocabulary
- **tokenizer_config.json** — Configuration metadata for the tokenizer
- **special_tokens_map.json** — Special tokens mapping (padding, unknown, etc.)

## Tokenizer Details

### Architecture

- **Type:** Byte-Pair Encoding (BPE)
- **Vocabulary Size:** 8,192 tokens
- **Training Data:** TinyStories dataset (learned from 160K stories)
- **Efficiency:** ~2x faster training compared to GPT-2 tokenizer

### Why Custom Tokenizer?

Standard tokenizers (like GPT-2's 50,257-token vocabulary) are overkill for children's stories. Our custom BPE tokenizer:
- **Reduces vocabulary** from 50,257 to 8,192 tokens
- **Speeds up training** by ~2x (28 min/epoch vs. 54 min/epoch)
- **Maintains quality** — perplexity difference: 18.78 (custom) vs. 17.12 (large model)
- **Optimizes for domain** — vocabulary learned specifically from story corpus

## How to Use

### Load the Tokenizer

```python
from transformers import PreTrainedTokenizerFast

# Load from this directory
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer/tokenizer.json",
    additional_special_tokens=[...]
)

# Tokenize text
tokens = tokenizer.encode("The princess smiled at the castle.")
print(tokens)  # [487, 234, 891, 123, ...]

# Decode tokens back to text
text = tokenizer.decode(tokens)
print(text)  # "The princess smiled at the castle."
```

### In the Notebook

The notebook automatically loads this tokenizer when `custom_tokenizer = True`:

```python
custom_tokenizer = True

if custom_tokenizer == True:
    # Load the custom BPE tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer/tokenizer.json")
    vocab_size = tokenizer.vocab_size
```

## Tokenizer Statistics

- **Vocabulary Size:** 8,192
- **Training Corpus:** 160,000 TinyStories
- **Token Frequency Range:** High-frequency tokens (common words) to low-frequency tokens (rare words)
- **Special Tokens:** Standard Hugging Face tokens (padding, unknown, end-of-sequence)

## Reproducibility

The tokenizer configuration is frozen and documented in:
- `tokenizer_config.json` — All settings for consistent loading
- `special_tokens_map.json` — Special token definitions

This ensures that if you train a new model with this tokenizer, the token assignments remain consistent and reproducible.

## Comparison: Custom BPE vs. GPT-2 Tokenizer

| Aspect | Custom BPE | GPT-2 |
|---|---|---|
| Vocabulary Size | 8,192 | 50,257 |
| Training Time/Epoch | 28 min | 54 min |
| Model Size | ~2.5M params | ~2.5M params |
| Val. Perplexity | 18.78 | 17.12* |
| Domain Specificity | Story-optimized | General-purpose |

*Note: The large model with GPT-2 tokenizer had more parameters and better resources. The custom tokenizer achieves competitive results with far better efficiency.

## Citation

If you use this tokenizer in your work, please cite:

```bibtex
@misc{chiari2025storygeneration,
  title={Conditional Story Generation with GPT-2},
  author={Chiari, Niccolò and Gardenghi, Alessandro and Grillini, Federico},
  year={2025},
  school={University of Bologna}
}
```
