# Causal Explanations using Statistical Fault Localisation

This is a hybrid XAI (Explainable AI) framework that draws inspiration from RISE (https://arxiv.org/abs/1806.07421) in the creation of Binary Masks as a method of perturbing the input of images, and uses Statistical Fault Localization (https://arxiv.org/pdf/1908.02374) as a pixel relevance metric. The output is a Saliency Map highlighting the most relevant pixels for a classification. This framework serves as a localized technique, capable of providing deep explanations for Image Classifiers on a single class. While slower than other XAI techniques, it achieves better results with a smaller GPU footprint.

To measure the effectiveness of my implementation, I used the Causal Metrics introduced in RISE and an implementation of Tristan's saliency maps metric in https://arxiv.org/abs/2201.13291


<p align="center">
  <img src=https://github.com/eaguaida/causal-explainer/blob/main/images/explainer_blueprint.png?raw=true />
</p>

# How to use it?
### To explain a single image:
```sh
python explainer.py path_to_image
```
### To explain a batch of images:
```sh
python explainer.py path_to_folder
```
### Fault Localisation 

<p align="center">
  <img src=https://github.com/eaguaida/RISE-SFL/blob/main/images/formulas.png?raw=true />
</p>
