import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix  # evaluation metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # one level up from src/
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

modelname = input("Enter the name of the saved model file (without extension, default 'logreg_model'): ").strip()
if not modelname:
    modelname = "logreg_model"
vectorizername = input("Enter the name of the saved vectorizer file (without extension, default 'vectorizer'): ").strip()
if not vectorizername:  
    vectorizername = "vectorizer"

filename = input("Enter the name of the CSV file to explore (without extension, default 'dataset'): ").strip()

DATA_PATH = os.getenv("PROCESSED_DATA_DEST_PATH")
if not filename:
    filename = "dataset"
df = pd.read_csv(os.path.join(DATA_PATH, f"{filename}.csv"), dtype={'month': str})

model = joblib.load(f"models/pred/{modelname}.pkl")
vectorizer = joblib.load(f"models/vector/{vectorizername}.pkl")

# Visualize articles by month
month_counts = df.groupby(['year', 'month'])['headline'].count().reset_index()
sns.barplot(x='year', y='headline', hue='month', data=month_counts)
plt.xlabel('Year')
plt.ylabel('Number of articles')
plt.title('Number of articles by month')
plt.show()

# Visualize model using confusion matrix
'''
y_pred = model.predict(vectorizer.transform(df['headline']))
y_pred = y_pred.astype(str)
cm = confusion_matrix(df['organization'].astype(str), y_pred)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
'''