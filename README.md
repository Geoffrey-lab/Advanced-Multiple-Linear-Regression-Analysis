# Advanced Multiple Linear Regression Analysis

This repository contains a Jupyter Notebook that delves into advanced multiple linear regression analysis using the `mtcars` dataset. The notebook demonstrates detailed steps for data exploration, checking for linearity and multicollinearity, model fitting using `statsmodels`, and comprehensive model evaluation.

## Notebook Content Overview

### 1. Introduction
- Overview of multiple linear regression analysis.
- Loading and exploring the `mtcars` dataset.

### 2. Data Exploration
- Loading the dataset from a CSV file.
- Displaying the first 10 rows of the dataset.
- Descriptive statistics for the dataset.

### 3. Checking for Linearity
- Creating scatter plots to visually inspect the relationship between each predictor and the response variable (miles per gallon).

### 4. Checking for Multicollinearity
- Generating pairwise scatter plots for predictors.
- Creating a correlation heatmap to identify strong correlations among predictors.

### 5. Fitting the Model Using `statsmodels.OLS`
- Constructing the regression formula.
- Fitting the model using Ordinary Least Squares (OLS) method.
- Printing and interpreting the model summary.

## Key Code Snippets

### Data Loading and Preparation
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Data/regression_sprint/mtcars.csv', index_col=0)
df.head(10)
```

### Data Exploration
```python
df.describe().T
```

### Checking for Linearity
```python
fig, axs = plt.subplots(2, 5, figsize=(14,6))
fig.subplots_adjust(hspace=0.5, wspace=0.2)
axs = axs.ravel()

for index, column in enumerate(df.columns):
    axs[index-1].set_title(f"{column} vs. mpg", fontsize=16)
    axs[index-1].scatter(x=df[column], y=df['mpg'], color='blue', edgecolor='k')

fig.tight_layout(pad=1)
```

### Checking for Multicollinearity
```python
from seaborn import pairplot

# Generate pairwise scatter plots
g = pairplot(df.drop('mpg', axis='columns'))
g.fig.set_size_inches(9, 9)
```

```python
# Correlation heatmap
corr = df.drop('mpg', axis='columns').corr()
from statsmodels.graphics.correlation import plot_corr
fig = plot_corr(corr, xnames=corr.columns)
```

### Fitting the Model Using `statsmodels.OLS`
```python
import statsmodels.formula.api as sm

# Generating the regression formula
formula_str = 'mpg ~ ' + ' + '.join(df.columns[1:])
print(formula_str)

# Construct and fit the model
model = sm.ols(formula=formula_str, data=df)
fitted = model.fit()

# Print model summary
print(fitted.summary())
```

## Conclusion
This notebook provides a thorough guide to performing advanced multiple linear regression analysis using Python's `statsmodels` library. It covers data exploration, linearity checks, multicollinearity assessment, and model fitting, helping you understand and apply these techniques to your own datasets. Explore the notebook, experiment with the methods, and gain deeper insights into regression analysis. Contributions and feedback are welcome!

## Repository Structure
- **notebooks/**: Directory containing the Jupyter Notebook for the analysis.
- **data/**: Directory containing the dataset (if applicable).
- **README.md**: Overview of the repository and instructions.

Feel free to explore, fork, and contribute to this repository!
