# Pretrained Model Weights

This folder contains (or should contain) the pretrained GPT-2 model weights for the story generation task.

## Model Details

- **Filename:** `modello_piccolo_voc8192_ep2_256_2_4.pth`
- **Architecture:** 2-layer Transformer with:
  - 256-dimensional embeddings
  - 4 attention heads
  - 8,192 vocabulary size
- **Training:** 2 epochs on TinyStories dataset (160K samples)
- **File Size:** ~16 MB
- **Format:** PyTorch `.pth` format

## How to Use

### Loading the Model

```python
import torch
from transformers import GPT2Config

# Load model weights
model = GPT2Config(
    vocab_size=8192,
    n_positions=512,
    n_embd=256,
    n_layer=2,
    n_head=4,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
).to_model()

state_dict = torch.load('models/modello_piccolo_voc8192_ep2_256_2_4.pth')
model.load_state_dict(state_dict)
model.eval()
```

Or see the Jupyter notebook for a complete example.

## Obtaining the Weights

Due to GitHub's file size limitations (max 100MB per file, max 2GB per repo), the model weights may not be stored in this repository.

To obtain the pretrained weights:
1. Contact the project authors
2. Download from project documentation or shared storage
3. Place the `.pth` file in this `models/` directory before running the notebook

## Training Information

If you wish to retrain the model from scratch:

1. Run the notebook: `notebooks/Final_Project_ML.ipynb`
2. Set `load_pretrained=False` in the configuration section
3. The notebook will automatically train a new model
4. Save your trained weights using PyTorch: `torch.save(model.state_dict(), 'path/to/model.pth')`

## Model Performance

**Validation Metrics:**
- Cross-Entropy Loss: 2.93
- Perplexity: 18.78
- Training Time: ~56 minutes per epoch on GPU

**Generation Quality:**
- Feature-conditioned story generation with semantic coherence
- Token uniqueness rate: ~0.42 (good diversity)
- Average story length: 200-400 tokens

## Citation

If you use this pretrained model in your research, please cite:

```bibtex
@misc{chiari2025storygeneration,
  title={Conditional Story Generation with GPT-2},
  author={Chiari, Niccolò and Gardenghi, Alessandro and Grillini, Federico},
  year={2025},
  school={University of Bologna}
}
```
