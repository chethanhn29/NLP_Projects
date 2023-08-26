# SMS Spam Classification Project
[Kaggle link](https://www.kaggle.com/code/chethuhn/sms-spam-classifier1)
## Overview

This project involves the analysis of an SMS spam collection dataset. The main objective is to develop a classification model to distinguish between spam and ham (non-spam) text messages. The project includes the following steps:

## Step 1: Data Loading and Preprocessing

- Load the dataset from a CSV file using the Pandas library.
- Drop unnecessary columns and rename the remaining columns to 'label' and 'message'.
- Check the dataset for class balance.

## Step 2: Natural Language Processing (NLP) Setup

- Import the NLTK library for NLP tasks.
- Import NLTK's stopwords, stemming, and lemmatization tools.
- Download the WordNet corpus for lemmatization.
- Define a function `preprocess_text` for text preprocessing, allowing for stemming and lemmatization.

## Step 3: Text Preprocessing

- Apply the `preprocess_text` function to the text messages in the dataset, creating two versions: one with stemming and one with lemmatization.

## Step 4: Text Representation Models

- Prepare two text representation models: Bag of Words (BOW) and Term Frequency-Inverse Document Frequency (TF-IDF).
- The `Text_representation_models` function converts text data into numerical representations using these models.

## Step 5: Target Encoding

- Convert the target column ('label') from text format to numerical representation using one-hot encoding (1 for 'spam', 0 for 'ham').

## Step 6: Model Selection and Evaluation

- Import various classification models, including Logistic Regression, Decision Trees, Random Forest, Support Vector Machines, Naive Bayes, and more.
- Define functions for splitting data and evaluating model performance.
- Define a function `find_best_model` to iterate over different combinations of text preprocessing and representation techniques along with various models. It selects the best model based on F1 scores and other metrics.

## Step 7: Model Evaluation and Selection

- Execute the `find_best_model` function to identify the best model.
- Print model performance metrics such as accuracy, F1 scores, precision, recall, and confusion matrices for each model and preprocessing combination.
- Select the best model based on the F1 score.

## Step 8: Further Optimization

- Suggest methods to improve model performance, including parameter tuning, regularization techniques, handling imbalanced data, cross-validation, and exploring alternative text representation models like word2vec and n-grams.

This project aims to identify the best-performing model for classifying SMS messages as spam or ham based on various combinations of text preprocessing techniques and classification models. The final choice is MLP Classifier with lemmatization and TF-IDF as preprocessing and text representation techniques, respectively.

