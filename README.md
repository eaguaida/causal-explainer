# Causal Framework 


## Explaning Image Classifiers

# How to use it?
### To explain a single image:
```sh
python explainer.py path_to_image
```
### To explain a batch of images:
```sh
python explainer.py path_to_folder
```

## Classification:

| Label | Ochiai | Zoltar | Tarantula | Wong-1 |
|----------|-------|---------|-------------|--------|
| Dog | ![](https://github.com/eaguaida/causal-explainer/blob/main/images/ochiai_dog.png?raw=true) | ![](https://github.com/eaguaida/causal-explainer/blob/main/images/zoltar_dog.png?raw=true) | ![](https://github.com/eaguaida/causal-explainer/blob/main/images/tarantula_dog.png?raw=true) | ![](https://github.com/eaguaida/causal-explainer/blob/main/images/wong1_dog.png?raw=true) |
| Cat | ![](https://github.com/eaguaida/causal-explainer/blob/main/images/cat_ochiai.png?raw=true) | ![](https://github.com/eaguaida/causal-explainer/blob/main/images/cat_zoltar.png?raw=true) | ![](https://github.com/eaguaida/causal-explainer/blob/main/images/cat_tarantula.png?raw=true) | ![](https://github.com/eaguaida/causal-explainer/blob/main/images/cat_wong1.png?raw=true) |

Disclosure:

This algorithm is an individual project that is not yet as powerful as SHAP, Grad-CAM or LIME. However, the plan is to achieve this in the future.
