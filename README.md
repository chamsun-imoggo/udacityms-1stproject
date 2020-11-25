# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This [dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) is related with direct phone call marketing campaigns of a Portuguese banking institution to predict whether people subscribe the bank term deposit or not.

Here, we will test two appraoches by using Microsoft Azure Machine Learning platform. 
1) Use [HyperDrive](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters) which automates the hyperparameter tuning for Logistic Regression model 
2) Use [AutoML](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train)

## Scikit-learn Pipeline
Create the Scikit-learn Logistic Regression model and apply the HyperDrive hyperparameter tuning to auto-tune 'C' which is the inverse of regularization strength. More smaller, stronger regularization.

[Random sampling](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#random-sampling) supports discrete and continuous hyperparameters. It supports early termination of low-performance runs. 

[Bandit policy](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#bandit-policy) is based on slack factor/slack amount and evaluation interval. Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run.

## AutoML
The best AutoML pipeline is VotingEnsemble and hyperparameters are automatically controlled by AutoML since AutoML does data preprocessing, model selection and hyperparameter optimization, which is the end to end automated machine learning lifecycle solution.

## Pipeline comparison
The accuracy of Scikit-learn logistic regression model with HyperDrive is about 0.90996 and the accuracy of AutoML is about 0.91684 so the difference is not big in terms of accuracy.


## Future work
We can consider auto hyperparameter tuning for other hyperparameters such as max_iter, the maximum number of iterations taken for the solvers to converge or other hyperparameters metioned from the [scikit-learn logistic regression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

Or we can refer to the result of AutoML and try different algorithms with HyperDrive.

# udacityms-1stproject
