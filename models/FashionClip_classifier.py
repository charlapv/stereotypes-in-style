{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "85356979-9b3c-4083-a182-cd5d138348f0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.9256666666666666\n",
      "\n",
      "Confusion matrix:\n",
      " [[3832  466]\n",
      " [ 426 7276]]\n",
      "\n",
      "Classification report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.90      0.89      0.90      4298\n",
      "           1       0.94      0.94      0.94      7702\n",
      "\n",
      "    accuracy                           0.93     12000\n",
      "   macro avg       0.92      0.92      0.92     12000\n",
      "weighted avg       0.93      0.93      0.93     12000\n",
      "\n",
      "\n",
      "Results saved:\n",
      "fashionclip_predictions_60k.csv\n",
      "fashionclip_metrics_60k.json\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score\n",
    "import json\n",
    "\n",
    "df = pd.read_csv(\"fashionclip_embeddings_60k.csv\")\n",
    "\n",
    "X = df.drop(columns=[\"image_path\",\"gender\"])\n",
    "y = df[\"gender\"].map({\"male\":0,\"female\":1})\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X,\n",
    "    y,\n",
    "    test_size=0.2,\n",
    "    stratify=y,\n",
    "    random_state=42\n",
    ")\n",
    "\n",
    "clf = LogisticRegression(max_iter=2000)\n",
    "clf.fit(X_train, y_train)\n",
    "\n",
    "pred = clf.predict(X_test)\n",
    "\n",
    "# metrics\n",
    "acc = accuracy_score(y_test, pred)\n",
    "cm = confusion_matrix(y_test, pred)\n",
    "report = classification_report(y_test, pred, output_dict=True)\n",
    "\n",
    "print(\"Accuracy:\", acc)\n",
    "print(\"\\nConfusion matrix:\\n\", cm)\n",
    "print(\"\\nClassification report:\\n\", classification_report(y_test, pred))\n",
    "\n",
    "# Save predictions\n",
    "results = pd.DataFrame({\n",
    "    \"true_label\": y_test,\n",
    "    \"predicted_label\": pred\n",
    "})\n",
    "\n",
    "results.to_csv(\"fashionclip_predictions_60k.csv\", index=False)\n",
    "\n",
    "# Save metrics\n",
    "metrics = {\n",
    "    \"accuracy\": acc,\n",
    "    \"confusion_matrix\": cm.tolist(),\n",
    "    \"classification_report\": report\n",
    "}\n",
    "\n",
    "with open(\"fashionclip_metrics_60k.json\", \"w\") as f:\n",
    "    json.dump(metrics, f, indent=4)\n",
    "\n",
    "print(\"\\nResults saved:\")\n",
    "print(\"fashionclip_predictions_60k.csv\")\n",
    "print(\"fashionclip_metrics_60k.json\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d57b2b71-d0d2-4d8c-b406-41c8503c617a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
