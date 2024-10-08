# Causal Framework 

## Method overview

This is a novel XAI framework used to explain Machine Learning models and DNNs. It is a unified approach combining [RISE](https://arxiv.org/abs/1806.07421), [DeepCover](https://arxiv.org/pdf/1908.02374), and [Spectrum-based Software Fault Localization](https://arxiv.org/abs/2206.08345). It is based on the idea that to achieve a good explanation, we need to create a "Test Suite," which is a balanced dataset consisting of correct/incorrect versions of the input that have been manipulated in various ways. We simply call these manipulated inputs "mutants."

We create this test suite by generating masks that are applied to the input. These masks are essentially collections of values ranging from 0 to 1. For example, for an image, when a mask is applied, it modifies the original input by multiplying the values of the pixels by the mask. After generating this dataset, we create 4 variables that count how each feature was perturbed (activated/non-activated), and then split the test suite into failing/passing mutants. We then proceed to input these variables into the SFL formulas to get the final score for each feature.

For Image Classifiers, we don't measure feature importance but pixel importance, so the output is a Saliency Map of the pixels that are most important for the model's prediction.

## Explaning Image Classifiers


### Use

Single image:
```sh
python explainer.py path_to_image
```
Batch of images:
```sh
python explainer.py path_to_folder
```
### How it works?

I've created a tutorial explaining ResNet50 [here](https://github.com/eaguaida/causal-explainer/blob/main/tutorial_resnet.ipynb).

<img src="https://github.com/eaguaida/causal-explainer/blob/main/images/explainer_blueprint.png?raw=true">

### Results
| Label | Ochiai | Zoltar | Tarantula | Wong-1 |
|----------|-------|---------|-------------|--------|
| Dog | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/ochiai_dog.png?raw=true" width="130" height="130"> | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/zoltar_dog.png?raw=true" width="130" height="130"> | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/tarantula_dog.png?raw=true" width="130" height="130"> | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/wong1_dog.png?raw=true" width="130" height="130"> |
| Cat | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/cat_ochiai.png?raw=true" width="130" height="130"> | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/cat_zoltar.png?raw=true" width="130" height="130"> | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/cat_tarantula.png?raw=true" width="130" height="130"> | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/cat_wong1.png?raw=true" width="130" height="130"> |

#

## Explaning Tabular Data

Models supported:  
- Tree based models (Decision Trees, Random Forest, XGBoost, etc)
- Linear Models (Linear, Logistic Regression, etc)
- MLP

### Iris Dataset:

In this example, we can explain the model's prediction on the Iris dataset. The code for this example can be found [here](https://github.com/eaguaida/causal-explainer/blob/main/tabular-explainer/iris_data_explanations.ipynb).

```sh
#Loading libraries
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from tabular_explainer import *
#Loading dataset
iris = datasets.load_iris()
iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#Train model
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
explanation = explain_tabular(clf, X_test)
```
Plotting only Wong-1 measure

```sh
plot_bar(explanation, view='Raw', measure='Wong1')
```
<img src="https://github.com/eaguaida/causal-explainer/blob/main/images/Wong1_raw_importance.png?raw=true" width="500" height="300">

Plotting all measures

```sh
plot_bar(explanation, view='Raw', measure='All')
```
| Ochiai | Zoltar | Tarantula | Wong-1 |
|-------|---------|-------------|--------|
| <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/Ochiai_raw_importance.png?raw=true" width="150" height="150"> | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/Zoltar_raw_importance.png?raw=true" width="150" height="150"> | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/Tarantula_raw_importance.png?raw=true" width="150" height="150"> | <img src="https://github.com/eaguaida/causal-explainer/blob/main/images/Wong1_raw_importance.png?raw=true" width="150" height="150"> |

### Diabetes Dataset:

While the performance of the explanation model is still experimental, it already achieves the purpose of demonstrating if a model is biased towards a specific feature. Here it's shown that the model is NOT biased towards the 'sex' feature, for example.
```sh
#Loading libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from tabular_explainer import *
#Loading dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
rforest = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=0)
rforest.fit(X_train, y_train)
explanation = explain_tabular(rforest, X_test)
plot_bar(explanation, view='Raw', measure='Wong1')
```
<img src="https://github.com/eaguaida/causal-explainer/blob/main/images/diabetes_wong1.png?raw=true" width="500" height="300">


