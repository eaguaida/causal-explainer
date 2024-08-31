# SpecFault - Explainability Framework


## Explaning Image Classifiers using 

# How to use it?
### To explain a single image:
```sh
python explainer.py path_to_image
```
### To explain a batch of images:
```sh
python explainer.py path_to_folder
```

## Classification

| Image | Ochiai | Zoltar | Tarantula | Wong-1 |
|----------|-------|---------|-------------|--------|
| Dog | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/ochiai_dog.png?raw=true" width="200" height="200"> | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/zoltar_dog.png?raw=true" width="200" height="200"> | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/tarantula_dog.png?raw=true" width="200" height="200"> | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/wong1_dog.png?raw=true" width="200" height="200"> |
| Cat | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/cat_ochiai.png?raw=true" width="200" height="200"> | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/cat_zoltar.png?raw=true" width="200" height="200"> | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/cat_tarantula.png?raw=true" width="200" height="200"> | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/cat_wong1.png?raw=true" width="200" height="200"> |



Supported Models:
Image Classifiers: VGG-16, ResNet
Machine Learning: Tree-based, linear and Neural Networks.
Disclosure:

This algorithm is an individual project that is not yet as powerful as SHAP, Grad-CAM or LIME. However, the plan is to achieve this in the future.
