import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json

df = pd.read_csv("/workspace/fusion_embeddings_60k.csv")

X = df.drop(columns=["image_path", "gender"])
y = df["gender"].map({"male": 0, "female": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

clf = LogisticRegression(max_iter=3000)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

acc = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred)
report_text = classification_report(y_test, pred)
report_dict = classification_report(y_test, pred, output_dict=True)

print("Accuracy:", acc)
print("\nConfusion matrix:\n", cm)
print("\nClassification report:\n", report_text)

pd.DataFrame({
    "true_label": y_test,
    "predicted_label": pred
}).to_csv("/workspace/fusion_predictions_60k.csv", index=False)

with open("/workspace/fusion_metrics_60k.json", "w") as f:
    json.dump({
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict
    }, f, indent=4)

print("Saved:")
print("/workspace/fusion_predictions_60k.csv")
print("/workspace/fusion_metrics_60k.json")