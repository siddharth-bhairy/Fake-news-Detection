# Fake-news-Detection
Project Overview
This project focuses on building a Fake News Detection model using a dataset of news articles. The model classifies news articles as either real or fake using Natural Language Processing (NLP) techniques and a Logistic Regression classifier. The dataset contains the title, author, and label of each news article.

Table of Contents
Project Overview
Project Structure
Dependencies
Dataset
Data Preprocessing
Model Building
Evaluation
Usage
Results

Dependencies
To run this project, you need the following libraries:

numpy
pandas
re (regular expressions)
nltk (Natural Language Toolkit)
sklearn (scikit-learn)

Dataset
The dataset (train copy.csv) consists of the following columns:

author: Name of the author of the news article
title: Title of the news article
label: 0 indicates real news, and 1 indicates fake news
Data Preprocessing
Handling Missing Values: Missing values in the dataset are filled with empty strings.
Text Merging: The author and title columns are merged to form a content column.
Text Cleaning & Stemming:
All non-alphabetical characters are removed.
The text is converted to lowercase.
Stopwords are removed, and words are stemmed using the PorterStemmer from NLTK.
Feature Extraction:
The cleaned text data is converted into numerical features using TF-IDF Vectorization.
Model Building
The project uses Logistic Regression from scikit-learn for classification.
The dataset is split into training (80%) and testing (20%) sets.
Evaluation
The model is evaluated using accuracy score on both the training and testing data.
