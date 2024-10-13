# Breast Cancer Wisconsin (Diagnostic) Dataset


# Motivation

The Breast Cancer Wisconsin (Diagnostic) Dataset aims to aid the research and development of machine learning techniques for improvement in the prediction of breast cancer.

The dataset was introduced in the paper Nuclear Feature Extraction For Breast Tumour Diagnosis (Street, Wolberg, Managsarin, 1993).


# Composition
The dataset consists of 30 features, listed in the table below.  The Diagnosis target column is a binary classification that can be used to train and test the models.

| VariableName       | 	Role                   | 	Type             |
|--------------------|-------------------------|-------------------|
| ID                 | 	ID                     | 	Categorical      |
| Diagnosis          | 	Target                 | 	Categorical      |
| radius1            | 	Feature                | 	Continuous       |
| texture1           | 	Feature                | 	Continuous       |
| perimeter1         | 	Feature                | 	Continuous       |
| area1              | 	Feature                | 	Continuous       |
| smoothness1        | 	Feature                | 	Continuous       |
| compactness1       | 	Feature                | 	Continuous       |
| concavity1         | 	Feature                | 	Continuous       |
| concave_points1    | 	Feature                | 	Continuous       |
| symmetry1          | 	Feature                | 	Continuous       |
| fractal_dimension1 | 	Feature                | 	Continuous       |
| radius2            | 	Feature                | 	Continuous       |
| texture2           | 	Feature                | 	Continuous       |
| perimeter2         | 	Feature                | 	Continuous       |
| area2              | 	Feature                | 	Continuous       |
| smoothness2        | 	Feature                | 	Continuous       |
| compactness2       | 	Feature                | 	Continuous       |
| concavity2         | 	Feature                | 	Continuous       |
| concave_points2    | 	Feature                | 	Continuous       |
| symmetry2          | 	Feature                | 	Continuous       |
| fractal_dimension2 | 	Feature                | 	Continuous       |
| radius3            | 	Feature                | 	Continuous       |
| texture3           | 	Feature                | 	Continuous       |
| perimeter3         | 	Feature                | 	Continuous       |
| area3              | 	Feature                | 	Continuous       |
| smoothness3        | 	Feature                | 	Continuous       |
| compactness3       | 	Feature                | 	Continuous       |
| concavity3         | 	Feature                | 	Continuous       |
| concave_points3    | 	Feature                | 	Continuous       |
| symmetry3          | 	Feature                | 	Continuous       |
| fractal_dimension3 | 	Feature                | 	Continuous       |


# Collection
The features were computed from a digitised image of a fine needle aspirate (FNA) taken from breast tissue.  Taken from Nuclear Feature Extraction For Breast Tumour Diagnosis (Street, Wolberg, Managsarin, 1993):

_With an interactive interface, the user initializes active contour models, known as snakes, near the boundaries of a set of cell nuclei.  The customized snakes are deformed to the exact shape of the nuclei.  Ten such features are computed for each nucleus, and the mean value, largest (or "worst") value and standard error of each feature are found over the range of the isolated cells._

A sample of this dataset was downloaded from Kaggle https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset in csv format with 

# Pre-processing
* The target column Diagnosis is dropped before training the model.
* The categorical labels are converted to numeric values (0=B for benign, 1=M for malignant).
* The features are standardised using StandardScaler module from the scikit-learn package, to calculate a standard score *z*

  *z = (x - u) / s*

    for *x*, given *u* mean and *s* standard deviation.


# Distribution 
The University of California, Irvine made this dataset https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic available in the UC Irvine Machine Learning Repository.

# Maintenance
The data is stored and maintained by UC Irvine Machine Learning Repository.

# Legal & Ethical
The data does not contain any information that could identify individuals whose tissue samples were used to extract the computed features.