import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns  #data visualization library
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix  # evaluation metrics

from sklearn.datasets import fetch_openml 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

import os
from dotenv import load_dotenv

# Resolve .env path relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # one level up from src/
ENV_PATH = os.path.join(BASE_DIR, ".env")



load_dotenv(dotenv_path=ENV_PATH)

DATA_PATH = os.getenv("PROCESSED_DATA_DEST_PATH")

# Ensure models directory exists
os.makedirs("models", exist_ok=True)
os.makedirs("models/pred", exist_ok=True)
os.makedirs("models/vector", exist_ok=True)

filename = input("Enter the name of the CSV file to explore (without extension, default 'dataset'): ").strip()

modelname = input("Enter the name for the saved model (without extension, default 'logreg_model'): ").strip()
if not modelname:
    modelname = "logreg_model"
vectorizername = input("Enter the name for the saved vectorizer file (without extension, default 'vectorizer'): ").strip()
if not vectorizername:
    vectorizername = "vectorizer"


if not filename:
    filename = "dataset"
df = pd.read_csv(os.path.join(DATA_PATH, f"{filename}.csv"), dtype={'month': str})

tempdf = df.copy()
tempdf.drop(columns=['year', 'date'], inplace=True) # Do not drop organization or fine_topic for now

TOP_TOPIC_LIMIT = 80
chosen_topic_label = 'organization'
top_topics = tempdf[chosen_topic_label].value_counts().nlargest(TOP_TOPIC_LIMIT).index
filtered_df = tempdf[tempdf[chosen_topic_label].isin(top_topics)]

final_df = filtered_df.copy()

y = final_df[chosen_topic_label]

vectorizer = TfidfVectorizer()
X_title = vectorizer.fit_transform(final_df['headline'])

X_month = StandardScaler().fit_transform(final_df[['month']]) * 0.01

#X = hstack([X_title, X_month])
X = X_title  # Currently not using month in training

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

#model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000, verbose=1, n_jobs=-1)
model = LogisticRegression(solver="saga", max_iter=1000, verbose=1, n_jobs=-1)
model.fit(X_train, y_train)

joblib.dump(model, f"models/pred/{modelname}.pkl")
joblib.dump(vectorizer, f"models/vector/{vectorizername}.pkl")
print("Model and vectorizer saved.")

y_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, y_pred)
print("Train accuracy:", train_acc)

y_pred_val = model.predict(X_test)
val_acc = accuracy_score(y_test, y_pred_val)
print("Validation accuracy:", val_acc)