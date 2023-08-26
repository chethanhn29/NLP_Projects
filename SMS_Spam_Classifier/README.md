# SMS Spam Detection Project

## Overview

This project is a demonstration of SMS spam detection using various text representation models and Naive Bayes classifiers. It explores different text preprocessing techniques, text representations (Bag of Words and TF-IDF), and Naive Bayes variants (Gaussian, Multinomial, Complement, Bernoulli) to classify SMS messages as either spam or ham (non-spam). The goal is to build a robust spam detection model and compare the performance of different approaches.

## Table of Contents

- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Text Representation](#text-representation)
- [Naive Bayes Models](#naive-bayes-models)
- [Evaluation Metrics](#evaluation-metrics)
- [Optimal Number of Features](#optimal-number-of-features)
- [Results](#results)
- [Best Model](#best-model)
- [Conclusion](#conclusion)

## Project Structure
- [code]
sms_spam_detection.ipynb (Jupyter Notebook with the code)
- [Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) (The dataset used for training and testing)
- README.md (This file)


## Data Preprocessing

- **Cleaning**: Non-alphabetic characters are removed from SMS messages.
- **Lowercasing**: All text is converted to lowercase for uniformity.
- **Tokenization**: SMS messages are tokenized into words.
- **Stopword Removal**: Common English stopwords are removed.
- **Stemming/Lemmatization**: Stemming or lemmatization is applied to reduce words to their base form.

## Text Representation

Two text representation models are explored:

- **Bag of Words (BOW)**: A basic model where each document is represented as a vector of word frequencies.
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: A more advanced model that considers the importance of words in a document relative to their frequency in the entire corpus.

## Naive Bayes Models

Several variants of the Naive Bayes classifier are implemented:

- **Gaussian Naive Bayes**: Suitable for continuous data.
- **Multinomial Naive Bayes**: Designed for count-based data like text.
- **Complement Naive Bayes**: Especially useful for imbalanced datasets.
- **Bernoulli Naive Bayes**: Applicable when features are binary (e.g., presence or absence of words).

## Evaluation Metrics

Various metrics are used to evaluate model performance:

- **Accuracy**: Overall correctness of the model.
- **Precision**: Proportion of true positive predictions out of all positive predictions.
- **Recall**: Proportion of true positive predictions out of all actual positive instances.
- **F1-score**: Harmonic mean of precision and recall, providing a balanced measure.
- **Confusion Matrix**: A table showing true positive, true negative, false positive, and false negative counts.

## Optimal Number of Features

An optimal number of top features (words) in the text representation is determined through experimentation. In this project, 6000 top features were found to work well for the dataset.

## Results

### Model Performance with Different Text Preprocessing Approaches

| Preprocessing      | Model                | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------------------|----------|-----------|--------|----------|
| No Preprocessing   | Gaussian Naive Bayes | 0.865    | 0.492     | 0.877  | 0.631    |
| Stemming           | Gaussian Naive Bayes | 0.865    | 0.979     | 0.864  | 0.918    |
| Lemmatization      | Gaussian Naive Bayes | 0.881    | 0.972     | 0.886  | 0.927    |
| No Preprocessing   | Gaussian Naive Bayes | 0.968    | 0.965     | 0.999  | 0.982    |

### Analysis

1. **No Preprocessing - Bag of Words (BOW)**: When using BOW with no preprocessing, the Gaussian Naive Bayes model achieved an accuracy of 0.865. It demonstrated decent precision and recall for identifying spam messages. The F1-score indicates a reasonably balanced trade-off between precision and recall.

2. **Stemming - Bag of Words (BOW)**: With stemming applied to BOW, the Gaussian Naive Bayes model improved its precision, recall, and F1-score, resulting in a higher F1-score of 0.918. This suggests that stemming benefited the model's performance.

3. **Lemmatization - Bag of Words (BOW)**: Lemmatization further improved the model's performance, with an accuracy of 0.881 and an F1-score of 0.927. It achieved a good balance between precision and recall.

4. **No Preprocessing - TF-IDF**: Without preprocessing using TF-IDF, the Gaussian Naive Bayes model achieved a remarkable accuracy of 0.968. It demonstrated high precision, recall, and F1-score, indicating that TF-IDF alone produced excellent results.

### Best Model

The best-performing model in this project is the Gaussian Naive Bayes model with TF-IDF text representation (accuracy: 0.968, F1-score: 0.982). This model outperforms others in both accuracy and F1-score. 

The high F1-score of 0.982 suggests that it achieves a strong balance between precision and recall. In this context, it is crucial to minimize false positives (classifying non-spam as spam) while maintaining high recall (capturing as many spam messages as possible). The Gaussian Naive Bayes model with TF-IDF achieves this balance effectively.

It's important to note that the choice of preprocessing and text representation can vary depending on the dataset and task. In this specific project, the TF-IDF representation with no preprocessing (except stop words removal) proved to be the most effective for SMS spam detection, likely due to its ability to capture term frequency-inverse document frequency information effectively.

Further experimentation and evaluation on different datasets may be necessary to confirm these results for other spam detection tasks.


## Conclusion

The project demonstrates the process of SMS spam detection using different text representation models and Naive Bayes classifiers. It concludes that the TF-IDF representation and the Bernoulli Naive Bayes model with 6000 top features provide the best results for this specific task. The README serves as a guide for understanding the project and its various components.
