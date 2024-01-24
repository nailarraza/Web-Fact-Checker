from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

app = Flask(__name__)

# Load data from CSV
data = pd.read_csv('Data_Training.csv')

# Pre-processing teks (contoh: lowercase, menghapus karakter khusus, stopwords)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Load data from CSV
data = pd.read_csv('Data_Training.csv')

# Pre-processing teks (contoh: lowercase, menghapus karakter khusus, stopwords)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = ' '.join([ps.stem(word) for word in text.split() if word not in stop_words])
    return text

data['narasi'] = data['narasi'].apply(preprocess_text)

# Vectorize data teks menggunakan TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_vectorized = vectorizer.fit_transform(data['narasi'])

# Train Naive Bayes model
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_vectorized.toarray(), data['label'])

# Train Random Forest model
random_forest_model = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=5, min_samples_leaf=1, random_state=42)
random_forest_model.fit(X_vectorized, data['label'])


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])

    # Predict using Naive Bayes
    nb_prediction = naive_bayes_model.predict(vectorized_text.toarray())

    # Predict using Random Forest
    rf_prediction = random_forest_model.predict(vectorized_text)

    return render_template('result.html', text=text, nb_prediction=nb_prediction, rf_prediction=rf_prediction)

if __name__ == '__main__':
    app.run(debug=True)
