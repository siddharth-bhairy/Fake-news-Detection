import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from wordcloud import WordCloud
import nltk

# nltk.download('stopwords')

# Data Pre-processing
news_dataset = pd.read_csv('train copy.csv')
news_dataset = news_dataset.fillna('')

# Merging the title and author name
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

# Separating the data and label
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']

# Stemming
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)

# Converting the text to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(news_dataset['content'].values)
X = vectorizer.transform(news_dataset['content'].values)
Y = news_dataset['label'].values

# Splitting the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Training the Model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluation
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the training data:', training_data_accuracy)
print('Accuracy score of the test data:', test_data_accuracy)

# 1. Class Distribution (Pie Chart)
plt.figure(figsize=(7, 7))
class_dist = news_dataset['label'].value_counts()
plt.pie(class_dist, labels=['Real', 'Fake'], autopct='%1.1f%%', colors=['#66b3ff', '#ff6666'], startangle=140, explode=(0.1, 0))
plt.title('Class Distribution (Real vs Fake News)', fontsize=14)
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
plt.show()

# 2. Accuracy Comparison Bar Chart
plt.figure(figsize=(8, 5))
accuracy_scores = [training_data_accuracy, test_data_accuracy]
labels = ['Training Accuracy', 'Test Accuracy']
bar_colors = ['#4CAF50', '#FF9800']  # Green for training, orange for test
plt.bar(labels, accuracy_scores, color=bar_colors)
plt.ylim(0.5, 1.0)
plt.title('Model Accuracy Comparison', fontsize=14)
plt.ylabel('Accuracy Score', fontsize=12)
plt.show()

# 3. Confusion Matrix Heatmap
cm = confusion_matrix(Y_test, X_test_prediction)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'], cbar=False)
plt.title('Confusion Matrix', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.show()

# 4. Word Cloud for Most Common Words
all_words = ' '.join([text for text in news_dataset['content']])
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='coolwarm', max_words=100, min_font_size=10).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words in the News Dataset', fontsize=16)
plt.show()
