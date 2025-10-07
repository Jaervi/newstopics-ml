from xml.parsers.expat import model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns  #data visualization library
from sklearn.metrics import accuracy_score, classification_report
import time


from sklearn.datasets import fetch_openml 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier


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

le = LabelEncoder()
y = le.fit_transform(final_df[chosen_topic_label])

vectorizer = TfidfVectorizer()
X_title = vectorizer.fit_transform(final_df['headline'])
X = X_title  # Currently not using month in training

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # 2 hidden layers: 100 and 50 neurons
    #hidden_layer_sizes=(512, 256, 128),
    activation='relu',
    solver='adam',
    max_iter=1,
    verbose=True,
    warm_start=True,
    random_state=42
)

train_accuracies = []
val_accuracies = []
times = []

for iteration in range(25):
    start = time.perf_counter()
    mlp.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    times.append(elapsed)

    y_pred = mlp.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred)

    y_pred_val = mlp.predict(X_test)
    val_acc = accuracy_score(y_test, y_pred_val)

    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"Iteration {iteration+1}: Accuracies = train: {train_acc:.4f}  val: {val_acc:.4f} (Time: {elapsed:.2f} seconds)")

print("Train accuracies over iterations:", train_accuracies)
print("Validation accuracies over iterations:", val_accuracies)

joblib.dump(mlp, f"models/pred/{modelname}.pkl")
joblib.dump(vectorizer, f"models/vector/{vectorizername}.pkl")
joblib.dump(le, f"models/vector/{vectorizername}_labelencoder.pkl")
print("Model and vectorizer saved.")
