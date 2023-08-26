# SMS Spam Classification Project

## Overview
This project aims to build a machine learning model for classifying SMS messages as either spam or ham (non-spam). The dataset used for this project is the "SMS Spam Collection Dataset."

## Steps Taken

1. **Data Loading and Exploration**:
   - Loaded the dataset using pandas and examined its structure.
   - Removed unnecessary columns and renamed the remaining ones for clarity.

2. **Data Imbalance Check**:
   - Checked the distribution of labels (spam and ham) in the dataset and found it to be imbalanced.

3. **Text Preprocessing**:
   - Performed text preprocessing, including tokenization, lowercasing, and removal of stopwords, using both stemming and lemmatization techniques.
   - Created three types of text data:
     - Raw Text
     - Stemmed Text
     - Lemmatized Text

4. **Text Representation**:
   - Utilized Word2Vec embeddings for text representation.
   - Explored various parameters of the Word2Vec model, such as vector size, window size, min_count, and the number of workers.
   - Obtained Word2Vec representations of the text data.

5. **Data Transformation**:
   - Transformed the target column from text format to numerical representation using one-hot encoding.

6. **Model Building and Evaluation**:
   - Built a variety of classification models to classify SMS messages.
   - Evaluated each model's performance using metrics like accuracy, F1 score (micro, macro, weighted), recall (micro, macro, weighted), precision (micro, macro, weighted), and confusion matrix.
   - Experimented with different classification models, including Logistic Regression, Decision Tree, Random Forest, Support Vector Machine, Bernoulli Naive Bayes, K-Nearest Neighbors, XGBoost, MLPClassifier, AdaBoost, Linear Discriminant Analysis, Quadratic Discriminant Analysis, Gaussian Process Classifier, Extra Trees Classifier, Ridge Classifier, and Passive Aggressive Classifier.

7. **Conclusion**:
   - Evaluated the models and selected the best-performing model based on the F1 score.
   - Provided insights into model performance and suggested further improvements or model fine-tuning.

This project is designed to classify SMS messages into spam or ham categories, which can be useful for building automated spam filters for text messages. Different preprocessing techniques and text representations were explored to improve model accuracy, and various classification algorithms were assessed to determine the most effective approach for this specific task. Further optimization and tuning can be done to achieve even better results.

