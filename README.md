# Causal Framework 

## Method overview

This is a novel XAI framework used to explain Machine Learning models and DNNs, it is an unified approach combining [RISE](https://arxiv.org/abs/1806.07421), [DeepCover](https://arxiv.org/pdf/1908.02374) and [Spectrum-based Software Fault Localization](https://arxiv.org/abs/2206.08345). It is based on the idea that to achieve a good explanation we need to create a "Test Suit", which is a balanced dataset consisting of correct/incorrect versions of the input that have been manipulated in any sort of way, we simply named "mutant" when referring to these manipulated inputs.

We create this test suit by generating masks that are applied on the input, these mask are basically a collection of values ranging from 0 to 1, Ex, for an image, when this mask is applied they modify the original input by multiplying the values of the pixels by the mask. After generating this dataset, we generated 4 variables that count for how each feature was perturbed (activated/non activated), to then split the test suit into failing/passing mutants. We then proceed to throw these variables into the SFL formulas to get the final score for each feature. 

For Image Classifiers, we don't measure features but pixels importance, so the output is a Saliency Map of the pixels that are most important for the model's prediction.

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


To generate a saliency map for model's prediction, RISE queries black-box model on multiple randomly masked versions of input.
After all the queries are done we average all the masks with respect to their scores to produce the final saliency map. The idea behind this is that whenever a mask preserves important parts of the image it gets higher score, and consequently has a higher weight in the sum.


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

In this example we can try to explain the model's prediction on the Iris dataset, the code can be found [here](https://github.com/eaguaida/causal-explainer/blob/main/tabular-explainer/iris_data_explanations.ipynb). 

```sh
#Loading libraries
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from tabular_explainer import *
#Loading dataset
iris = datasets.load_iris()
iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#Train model
clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
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
Disclosure:
