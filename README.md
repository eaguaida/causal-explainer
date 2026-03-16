# Causal Explainer

An explainability method for black-box image classifiers that adapts [Spectrum-Based Fault Localization](https://arxiv.org/abs/2206.08345) (SFL) — a technique from software testing — to identify which pixels drive a neural network's prediction.

The key insight: in software testing, SFL finds buggy code by analyzing which lines execute in passing vs. failing tests. Causal Explainer applies the same principle to images — it treats randomly masked image variants as test cases and the classifier's predictions as pass/fail outcomes, then uses SFL formulas to rank each pixel's causal contribution.

## How It Works

1. **Mask Generation** — Random binary masks are applied to the input image, producing masked variants (mutants)
2. **Pass/Fail Classification** — Each mutant is classified by the model; mutants that preserve the original prediction pass, others fail
3. **SFL Scoring** — Four fault localization formulas (Ochiai, Tarantula, Zoltar, Wong-1) score each pixel based on how often it appears in passing vs. failing mutants, weighted by prediction confidence

<img src="https://github.com/eaguaida/causal-explainer/blob/main/images/explainer_blueprint.png?raw=true">

## Installation

```bash
git clone https://github.com/eaguaida/causal-explainer.git
cd causal-explainer
pip install -e .
```

## Quick Start

```python
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from causal_explainer import SFL, RelevanceScore, SaliencyMapVisualizer, get_device

# Load a pretrained model
device = get_device()
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = nn.Sequential(model, nn.Softmax(dim=1))
model = model.eval().to(device)

# Generate masked variants for a single image
sfl = SFL(model, input_size=(224, 224))
masks, sampled_tensor = sfl.generate_mutants_batch(
    img_path="data_demo/catdog.png",
    N=100, s=8, p1=0.2, target_class=243  # 243 = bull mastiff
)

# Compute confidence scores for each mutant
confidence_scores = np.zeros(100)
with torch.no_grad():
    for i in range(100):
        output = model(sampled_tensor[i].unsqueeze(0))
        confidence_scores[i] = torch.max(output, dim=1).values.item() * 100

# Calculate SFL pixel-importance scores
rs = RelevanceScore()
pixel_dataset, ochiai, tarantula, zoltar, wong1 = rs.run(
    confidence_scores, sampled_tensor, masks, N=100
)

# Visualize the saliency map
viz = SaliencyMapVisualizer("data_demo/catdog.png")
viz.visualize_pixel_scores(pixel_dataset)
```

See [`examples/`](examples/) for complete notebooks covering single-image analysis, batch processing, and benchmark evaluation.

## CLI Usage

```bash
# Process a folder of images (saves saliency maps)
python -m causal_explainer data_demo/ 100

# With custom parameters
python -m causal_explainer data_demo/ 200 --s 8 --p1 0.2
```

## SFL Formulas

The method uses four SFL metrics from the fault localization literature:

<img src="https://github.com/eaguaida/causal-explainer/blob/main/images/formulas.png?raw=true">

Where for each pixel: **Ep** = visible in passing mutants, **Ef** = visible in failing mutants, **Np** = hidden in passing mutants, **Nf** = hidden in failing mutants.

## Results

| Label | Ochiai | Zoltar | Tarantula | Wong-1 |
|-------|--------|--------|-----------|--------|
| Dog | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/ochiai_dog.png?raw=true" width="130" height="130"> | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/zoltar_dog.png?raw=true" width="130" height="130"> | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/tarantula_dog.png?raw=true" width="130" height="130"> | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/wong1_dog.png?raw=true" width="130" height="130"> |
| Cat | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/cat_ochiai.png?raw=true" width="130" height="130"> | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/cat_zoltar.png?raw=true" width="130" height="130"> | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/cat_tarantula.png?raw=true" width="130" height="130"> | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/cat_wong1.png?raw=true" width="130" height="130"> |

## Package Structure

```
causal_explainer/
├── __init__.py              # Top-level exports
├── __main__.py              # CLI entry point (python -m causal_explainer)
├── utils.py                 # Image loading, preprocessing, device detection
├── explainer.py             # Batch processing pipeline and argument parsing
├── synset_words.txt         # ImageNet class labels
├── masker/
│   ├── generation.py        # SFL — single-image mutant generation
│   └── batch.py             # SFL_batch — multi-image batch processing
├── sfl/
│   ├── relevance_score.py   # RelevanceScore — SFL scoring pipeline
│   └── formulas/
│       ├── fault_localization_metrics.py  # Unified metric interface
│       ├── ochiai.py
│       ├── tarantula.py
│       ├── zoltar.py
│       └── wong1.py
├── visuals/
│   ├── plots.py             # SFLVisualizer — mutant inspection
│   └── saliency.py          # SaliencyMapVisualizer — heatmap rendering
└── benchmark/
    └── evaluation.py        # CausalMetric — deletion/insertion metrics
```

## References

- Petsiuk et al., [RISE: Randomized Input Sampling for Explanation of Black-box Models](https://arxiv.org/abs/1806.07421), BMVC 2018
- Sun et al., [Causality-Based Neural Network Repair](https://arxiv.org/abs/1908.02374), ASE 2019
- Wong et al., [A Survey on Software Fault Localization](https://arxiv.org/abs/2206.08345), TSE 2016

## License

MIT — see [LICENSE](LICENSE).
