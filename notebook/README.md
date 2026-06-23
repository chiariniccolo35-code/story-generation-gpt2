# Notebooks

This folder contains the Jupyter notebooks for the story generation project.

## Files

### Final_Project_ML.ipynb

Complete implementation of the GPT-2 model for conditional story generation.

**Contents:**
1. **Dataset Loading** — Load TinyStories from Hugging Face Datasets
2. **Data Exploration** — Analyze dataset structure and narrative features
3. **Tokenization** — Custom BPE tokenizer training and configuration
4. **Model Architecture** — Define GPT-2 Transformer architecture
5. **Training** — Train the model on TinyStories dataset
6. **Evaluation** — Compute validation metrics (cross-entropy, perplexity)
7. **Story Generation** — Generate conditioned stories with manual and automatic decoding

**How to Run:**

```bash
jupyter notebook Final_Project_ML.ipynb
```

The notebook can be run sequentially from top to bottom. Key parameters are clearly marked for modification:
- `custom_tokenizer` — Use custom BPE vs. GPT-2 tokenizer
- `sub_sample_size` — Training set size (default: 160K)
- `batch_size` — Training batch size (default: 16)
- `n_epochs` — Number of training epochs (default: 2)

**Expected Runtime:**
- ~56 minutes per epoch on GPU (with 160K samples)
- ~28 minutes per epoch with custom tokenizer (vs. 54 minutes with GPT-2 tokenizer)

**Outputs:**
- Trained model weights (`.pth` file)
- Validation metrics (cross-entropy, perplexity)
- Generated story examples with different feature conditions

**Requirements:**
- PyTorch 2.0+
- Transformers 4.30+
- Datasets 3.6+
- GPU recommended (can run on CPU but much slower)

See `requirements.txt` in the root directory for complete dependencies.
