import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score  # evaluation metrics

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

train_accuracies = []
val_accuracies = []

iterations = 20
startIteration = 2

for i in range(startIteration - 1, iterations + (startIteration - 1)):
    print(f"Iteration {i+1}/{iterations + (startIteration - 1)}")
    model = LogisticRegression(
        solver="saga",
        max_iter=i,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    train_acc = accuracy_score(y_train, y_pred_train)
    val_acc = accuracy_score(y_val, y_pred_val)
    
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

print("Train accuracy:", train_accuracies[-1])

print("Validation accuracy:", val_accuracies[-1])

print("Test accuracy:", accuracy_score(y_test, model.predict(X_test)))

joblib.dump(model, f"models/pred/{modelname}.pkl")
joblib.dump(vectorizer, f"models/vector/{vectorizername}.pkl")
print("Model and vectorizer saved.")

r = range(startIteration, len(val_accuracies) + startIteration)
plt.plot(r, val_accuracies, label="validation accuracy")
plt.plot(r, train_accuracies, label="train accuracy")
plt.xticks(r)
plt.xlabel("Iteration")
plt.ylabel("Validation accuracy")
plt.title(f"Logistic Regression with 80 organization topics, 15% test size, saga solver, {iterations} iterations")
plt.legend()
plt.show()
