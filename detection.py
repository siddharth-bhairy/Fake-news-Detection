import numpy as np
import pandas as pd
import re 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
#nltk.download('stopwords')
#print(stopwords.words('english'))
#data pre-processing
#loading the dataset into a pandas dataset

news_dataset=pd.read_csv('train copy.csv')

#print first 5 rows
#print(news_dataset.head())

#counting the number of missing values
#print(news_dataset.isnull().sum())

#replacing the null values with empty string
news_dataset=news_dataset.fillna('')

#megring the title and author name
news_dataset['content']= news_dataset['author']+' '+news_dataset['title']
#print(news_dataset['content'])

#separating the data and label
X=news_dataset.drop(columns='label',axis=1)
Y=news_dataset['label']

# print(X)
# print(Y)

#stemming
port_stem=PorterStemmer()

def stemming(content):
    stemmed_content=re.sub('[^a-zA-Z]',' ',content)
    stemmed_content=stemmed_content.lower()
    stemmed_content=stemmed_content.split()
    stemmed_content=[port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content=' '.join(stemmed_content)
    return stemmed_content

news_dataset['content']=news_dataset['content'].apply(stemming)
# print(news_dataset['content'])


#separating the data and label
X=news_dataset['content'].values
Y=news_dataset['label'].values

# print(X)
# print(Y)

#converting the text to numerical data
vectorizer=TfidfVectorizer()
vectorizer.fit(X)
X=vectorizer.transform(X)

#print(X)

#splitting the dataset into training and test data

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

#training the Model : logistic regression model

model=LogisticRegression()
model.fit(X_train,Y_train)

#evaluation
#accuracy scare on training data

X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)

print('Accuracy score of the training data  : ',training_data_accuracy)

#accuracy scare on test data

X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)

print('Accuracy score of the training data  : ',test_data_accuracy)


#making a predictive system

X_new=X_test[0]
prediction=model.predict(X_new)

print(prediction)

if prediction==0:
    print("the news is real")
else:
    print('the news is Fake')
    
print(Y_test[0])