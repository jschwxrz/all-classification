# Acute Lymphoblastic Leukemia (ALL) Classification

## Project Overview

The machine learning problem at hand is the classification of acute lymphoblastic leukemia (ALL), based on microscopic images of blood cells. ALL is the most common type of childhood cancer, accounting for approximately 25% of pediatric cancers.

#### Input:

- A 224 by 224 RGB image of a blood cell

#### Output:

- 0 or 1, indicating what type of cell it is (HEM or ALL).

#### Evaluation Metric:

To evaluate the performance of a model, I chose the F1 score as the metric. As the dataset classes are imbalanced, accuracy as the metric might be misleading, in case the model always favors the majority class. The F1 score balances precision (minimizing false positives) and recall (minimizing false negatives for ALL) and is suitable for binary classification. Especially in medicine, it is important to focus on low false negative rates, making the F1 score a crucial metric.

## Dataset

The dataset ["Leukemia Classification"](https://www.kaggle.com/datasets/andrewmvd/leukemia-classification/data) [1] consists of 15,135 images of blood cells from 118 patients, segmented from microscopic images of blood samples. The labels are ALL (Acute Lymphoblastic Leukemia) and HEM (normal cells). According to the dataset description, ground truth labels were annotated by an expert oncologist in order to ensure the high quality of the data. It also states that the data is representative of images in the real world because they contain some staining noise and illumination errors, although these errors have largely been fixed in the course of acquisition.

Example of a cell of type ALL:

![Image of ALL cell](https://github.com/jschwxrz/all-classification/blob/main/images/ALL.bmp?raw=true)

Example of a cell of type HEM:

![Image of HEM cell](https://github.com/jschwxrz/all-classification/blob/main/images/HEM.bmp?raw=true)

[1] Gupta, A., & Gupta, R. (2019). ALL Challenge dataset of ISBI 2019 [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/tcia.2019.dc64i46r

## Model

The model is based on the InceptionV3 architecture and makes use of transfer learning using the ImageNet weights. The top is not included, and all the layers are unfrozen and therefore trainable. A dense layer with 256 neurons, ReLU activation function, and the final output layer with a softmax activation function build the top of the model.

Following hyperparameters were tuned using Keras-tuner:

- L2 regularization
- L1 regularization
- Dropout rate in the dense layer
- Number of nodes in the dense layers
- Activation function

Due to computational constraints, the rest of the hyperparameters were initialized using common values for the given hyperparameter or looking at similar CNNs and adapting theirs.

## Performance

The final model achieved a training accuracy of 100%, a validation accuracy of 95.88%, and a test accuracy of 95.5%. A variance of 4%, as seen in the small gap between training and validation/testing scores, indicates a slight overfitting.

It is not perfectly balanced, and the model is still slightly favoring the majority class with an F1 score of 97 for ALL and 94 for HEM. However, in this case, the most important metric, besides the overall F1 score, is the false negative rate for the ALL class. This is crucial because false negatives for this class mean undetected immature leukemic blasts. However, recall (false negatives) for the ALL class is at 98% and therefore almost perfect.

## Setup

To see the model performance in the notebook format, open ALL-classification.ipynb on GitHub.

To run the code:

Clone the repository:

- `git clone https://github.com/jschwxrz/all-classification.git`

Install the requirements:

- `pip install -r requirements.txt`

Either open ALL-classification.ipynb to run the notebook or run:

- `python ALL-classification.py`

To use the model that was trained in ALL-classification.ipynb, download the model.keras file.
