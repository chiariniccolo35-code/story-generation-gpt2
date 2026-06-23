# Conditional Story Generation with GPT-2

A deep learning project implementing a Transformer-based language model (GPT-2) for generating children's stories conditioned on narrative features.

## Project Overview

This project develops a generative language model capable of creating coherent, feature-conditioned short stories. Given a set of narrative tags (e.g., Dialogue, Conflict, Twist), the model generates contextually appropriate stories that embody the specified features.

**Key Innovation:** The model is conditioned on story features by prepending feature tokens to the input sequence, allowing fine-grained control over narrative elements without architectural modifications.

## Dataset

- **Dataset:** TinyStories (Xu et al., 2023)
- **Size:** 2.7M stories (160K used for training)
- **Features:** 6 narrative tags: BadEnding, Conflict, Dialogue, Foreshadowing, MoralValue, Twist

## Model Architecture

- **Base Model:** Transformer Decoder (GPT-2 style)
- **Embeddings:** 256-dimensional
- **Layers:** 2 Transformer blocks
- **Attention Heads:** 4
- **Context Length:** 512 tokens
- **Total Parameters:** ~2.5M (lightweight, efficient design)
- **Training:** 2 epochs on 160K story samples
- **Hardware:** GPU-accelerated training (PyTorch)

## Tokenization

We implemented **Byte-Pair Encoding (BPE)** for adaptive vocabulary building:
- Vocabulary size: 8,192 (vs. 50,257 for GPT-2)
- **Efficiency gain:** 28 min/epoch (BPE) vs. 54 min/epoch (GPT-2 tokenizer)
- Vocabulary learned from training corpus, reducing redundant tokens
- Tokenizer files included in `tokenizer/` folder for reproducibility

## Hyperparameter Tuning

Systematic exploration of model hyperparameters to balance performance and efficiency:
- Fixed external parameters: batch size (16), max length (512), epochs (2)
- Tuned internal parameters: embedding dimension, layers, attention heads, vocabulary size
- Iterative approach: progressively reduced model size while maintaining performance

## Results

### Quantitative Evaluation

| Model Size | Val. Cross-Entropy | Perplexity |
|---|---|---|
| Large (30M) | 2.84 | 17.12 |
| Medium (8M) | 2.89 | 17.99 |
| **Final (2.5M)** | **2.93** | **18.78** |

The final lightweight model achieves competitive performance with significantly fewer parameters.

### Qualitative Analysis

- **Feature Control:** Model successfully generates stories respecting input feature tags
- **Coherence:** Stories maintain narrative structure and semantic consistency
- **Creativity:** Token uniqueness rate ~0.42 (good diversity without repetition)
- **Manual Decoding:** Supports both greedy and sampling-based generation strategies

## Project Highlights

1. **Custom BPE Tokenizer:** Designed and trained tokenizer optimized for story generation (2x training speedup)
2. **Feature Conditioning:** Novel token-prepending approach for multi-feature control
3. **Efficient Architecture:** Demonstrates that smaller models (2.5M params) can achieve strong results
4. **Complete Pipeline:** Dataset loading, preprocessing, training, evaluation, and inference
5. **Systematic Evaluation:** Both quantitative metrics (perplexity, cross-entropy) and qualitative analysis
6. **Reproducibility:** Trained model weights, tokenizer files, and complete documentation included

## Files & Directories

```
.
├── README.md                                      # This file
├── requirements.txt                               # Python dependencies
├── notebooks/
│   └── Final_Project_ML.ipynb                    # Complete implementation & results
├── tokenizer/
│   ├── tokenizer.json                            # BPE tokenizer weights
│   ├── tokenizer_config.json                     # Tokenizer configuration
│   └── special_tokens_map.json                   # Special tokens mapping
└── models/
    └── README.md                                  # Guide to downloading pretrained model
```

## Pretrained Model

The trained model weights are available upon request due to GitHub's file size limitations.

**Model Details:**
- **Architecture:** 2-layer Transformer, 256 embeddings, 4 attention heads
- **Vocabulary Size:** 8,192 tokens
- **Training Data:** 160K TinyStories
- **File Size:** ~16 MB
- **Format:** PyTorch `.pth`

To use the pretrained model:
1. Request the weights file: `modello_piccolo_voc8192_ep2_256_2_4.pth`
2. Place in `models/` directory
3. Load in notebook as shown in the Jupyter notebook

## How to Use

### Installation

```bash
pip install -r requirements.txt
```

### Running the Notebook

```bash
jupyter notebook notebooks/Final_Project_ML.ipynb
```

The notebook can be run end-to-end. Key sections:

1. **Dataset Loading** — Loads TinyStories from Hugging Face
2. **Tokenization** — Loads the custom BPE tokenizer
3. **Model Configuration** — Sets up GPT-2 architecture
4. **Training** — Trains model from scratch or loads pretrained weights
5. **Evaluation** — Computes validation metrics
6. **Generation** — Generates conditioned stories with multiple decoding strategies

### Story Generation Example

To generate a story with specific features:

```python
from final_project_ml import generate_story

# Generate story with feature conditioning
generated = generate_story(
    model=model, 
    device=device, 
    story_features=["Dialogue", "Conflict"],
    input_text="Once upon a time, ",
    max_length=400,
    auto=True  # Automatic generation
)

print(generated)
```

### Using the Custom Tokenizer

```python
from transformers import PreTrainedTokenizerFast

# Load the trained BPE tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer/tokenizer.json")

# Tokenize text
tokens = tokenizer.encode("The princess smiled.")
print(tokens)

# Decode tokens
text = tokenizer.decode(tokens)
print(text)
```

## Team

**Authors:** Niccolò Chiari, Alessandro Gardenghi, Federico Grillini

**Course:** Introduction to Machine Learning  
**University:** University of Bologna, 2024/25 Academic Year

## References

- Vaswani, A., et al. (2017). Attention is All You Need. *Advances in Neural Information Processing Systems (NeurIPS)*.
- Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners. OpenAI Blog.
- Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. *Proceedings of ACL*.
- Xu, T., et al. (2023). TinyStories: How Small Can Language Models Be and Still Speak Coherently? *arXiv preprint arXiv:2305.07759*.

## License

Educational project. Available for academic use and research purposes.

---

## Getting Started

1. Clone or download this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Open the Jupyter notebook: `jupyter notebook notebooks/Final_Project_ML.ipynb`
4. Run cells sequentially to train, evaluate, and generate stories
5. For pretrained model weights, contact the authors or request from the repository

For questions or suggestions, feel free to open an issue or contact the authors.
