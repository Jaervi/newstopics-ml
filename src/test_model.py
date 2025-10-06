import joblib
import numpy as np

# Load model and vectorizer

modelname = input("Enter the name of the saved model file (without extension, default 'logreg_model'): ").strip()
if not modelname:
    modelname = "logreg_model"
vectorizername = input("Enter the name of the saved vectorizer file (without extension, default 'vectorizer'): ").strip()
if not vectorizername:  
    vectorizername = "vectorizer"

model = joblib.load(f"models/pred/{modelname}.pkl")
vectorizer = joblib.load(f"models/vector/{vectorizername}.pkl")

print("Test predictor loaded.")
print("Type 'exit' at any time to quit.\n")

while True:
    title = input("Enter a title: ")
    if title.lower() == "exit":
        break
    title = title.lower().replace(",", "").replace("ä", "a").replace("ö", "o").replace("-", " ").replace(":", "").strip()

    #month_str = input("Enter a month (1–12): ")
    #if month_str.lower() == "exit":
    #    break
    #
    #try:
    #    month = int(month_str)
    #    if not 1 <= month <= 12:
    #        print("⚠️ Please enter a valid month (1–12).")
    #        continue
    #except ValueError:
    #    print("⚠️ Please enter a number for the month.")
    #    continue

    # Preprocess
    X_title = vectorizer.transform([title])
    #X = np.hstack([X_title.toarray(), np.array([[month]])])
    X = X_title  # Currently not using month in prediction

    # Predict
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    print(f"Formatted title: {title}")
    
    if hasattr(model, 'coef_'):
        print(f"Predicted topic: {prediction}")
        print(f"Prediction confidence: {np.max(proba):.3f}")
        print(f"Importance of features (month is last): {model.coef_[model.classes_ == prediction]}")
        print(f"Top 10 topic probabilities:")
        top10_indices = np.argsort(proba)[-10:][::-1]
        for idx in top10_indices:
            print(f" - {model.classes_[idx]}: {proba[idx]:.3f}")
    else:
        le = joblib.load(f"models/vector/{vectorizername}_labelencoder.pkl")
        predicted_label = le.inverse_transform([prediction])[0]
        print(f"Predicted topic: {predicted_label}")
        print(f"Prediction confidence: {np.max(proba):.3f}")
        print(f"Top 10 topic probabilities:")
        top10_indices = np.argsort(proba)[-10:][::-1]
        for idx in top10_indices:
            label = le.inverse_transform([idx])[0]
            print(f" - {label}: {proba[idx]:.3f}")

    #print(f"   Probabilities: {dict(zip(model.classes_, proba.round(3)))}\n")
