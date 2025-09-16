import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

filename = input("Enter the name of the CSV file to explore (without extension, default 'dataset'): ").strip()

if not filename:
    filename = "dataset"
df = pd.read_csv(os.path.join(DATA_PATH, f"{filename}.csv"), dtype={'date': str})

df.drop(columns=['organization', 'fine_topic'], inplace=True)

df = df[:20000]

y = df['topic']
X_title = TfidfVectorizer().fit_transform(df['headline'])

X_month = StandardScaler().fit_transform(df[['date']])

X = hstack([X_title, X_month])

model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
model.fit(X, y)

y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)
print(acc)