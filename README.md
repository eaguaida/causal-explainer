# Causal Explainer

A black-box XAI framework for image classifiers combining [RISE](https://arxiv.org/abs/1806.07421) mask sampling with [Spectrum-based Fault Localization](https://arxiv.org/abs/2206.08345) (SFL) pixel ranking.

## Method Overview

The framework generates saliency maps by:

1. **Mask Generation** — Random binary masks (RISE-style) are applied to the input image to create "mutants"
2. **Test Suite Construction** — Mutants are classified as passing (correct prediction) or failing (incorrect prediction)
3. **SFL Scoring** — Four fault localization formulas (Ochiai, Tarantula, Zoltar, Wong-1) rank each pixel's importance based on its activation pattern across passing/failing mutants

<img src="https://github.com/eaguaida/causal-explainer/blob/main/images/explainer_blueprint.png?raw=true">

## Installation

```bash
git clone https://github.com/eaguaida/causal-explainer.git
cd causal-explainer
pip install -r requirements.txt
```

## Quick Start

```python
import torch
import torch.nn as nn
import torchvision.models as models
from causal_explainer import SFL, RelevanceScore, SaliencyMapVisualizer, get_device

# Load model
device = get_device()
model = models.resnet50(pretrained=True)
model = nn.Sequential(model, nn.Softmax(dim=1))
model = model.eval().to(device)

# Generate mutants for a single image
sfl = SFL(model, input_size=(224, 224))
masks, sampled_tensor = sfl.generate_mutants_batch(
    img_path="data_demo/catdog.png",
    N=100, s=8, p1=0.2, target_class=243
)

# Calculate SFL scores
rs = RelevanceScore()
# ... compute confidence scores, then:
# pixel_dataset, ochiai, tarantula, zoltar, wong1 = rs.run(confidence_scores, sampled_tensor, masks, N)

# Visualize
viz = SaliencyMapVisualizer("data_demo/catdog.png")
# viz.visualize_pixel_scores(pixel_dataset)
```

## CLI Usage

```bash
# Process a folder of images (saves saliency maps)
python -m causal_explainer.explainer data_demo/ 100

# With custom parameters
python -m causal_explainer.explainer data_demo/ 200 --s 8 --p1 0.2
```

## SFL Formulas

The framework implements four SFL metrics from the fault localization literature:

<img src="https://github.com/eaguaida/causal-explainer/blob/main/images/formulas.png?raw=true">

Where for each pixel: **Ep** = executed in passing tests, **Ef** = executed in failing tests, **Np** = not executed in passing tests, **Nf** = not executed in failing tests.

## Results

| Label | Ochiai | Zoltar | Tarantula | Wong-1 |
|-------|--------|--------|-----------|--------|
| Dog | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/ochiai_dog.png?raw=true" width="130" height="130"> | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/zoltar_dog.png?raw=true" width="130" height="130"> | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/tarantula_dog.png?raw=true" width="130" height="130"> | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/wong1_dog.png?raw=true" width="130" height="130"> |
| Cat | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/cat_ochiai.png?raw=true" width="130" height="130"> | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/cat_zoltar.png?raw=true" width="130" height="130"> | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/cat_tarantula.png?raw=true" width="130" height="130"> | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/cat_wong1.png?raw=true" width="130" height="130"> |

## Package Structure

```
causal_explainer/
├── __init__.py            # Top-level exports
├── utils.py               # Image loading, preprocessing, device detection
├── explainer.py           # CLI entry point
├── synset_words.txt       # ImageNet class labels
├── masker/
│   ├── generation.py      # SFL — single-image mutant generation
│   └── batch.py           # SFL_batch — multi-image batch processing
├── sfl/
│   ├── relevance_score.py # RelevanceScore — SFL algorithm pipeline
│   └── formulas/          # Ochiai, Tarantula, Zoltar, Wong-1
├── visuals/
│   ├── plots.py           # SFLVisualizer — mutant inspection
│   └── saliency.py        # SaliencyMapVisualizer — heatmap rendering
└── benchmark/
    └── evaluation.py      # CausalMetric — deletion/insertion metrics
```

## References

- Petsiuk et al., [RISE: Randomized Input Sampling for Explanation of Black-box Models](https://arxiv.org/abs/1806.07421), BMVC 2018
- Sun et al., [DeepCover: Structural Test Criteria for Deep Neural Networks](https://arxiv.org/abs/1908.02374), ASE 2019
- Wong et al., [A Survey on Software Fault Localization](https://arxiv.org/abs/2206.08345), TSE 2016

## License

See [LICENSE](LICENSE).
