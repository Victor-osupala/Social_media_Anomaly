import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load dataset
df = pd.read_csv("social_media_anomaly.csv")  # Replace with your dataset path
numeric_df = df.select_dtypes(include=[np.number]).dropna()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)

# Train Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(X_scaled)

# Predict anomaly labels and scores
# -1 = anomaly, 1 = normal
preds_raw = model.predict(X_scaled)
preds = np.where(preds_raw == -1, 1, 0)
scores = -model.decision_function(X_scaled)  # Higher scores = more anomalous

df['anomaly'] = preds

# Use ground truth if exists; else use preds as dummy true labels
true_labels = df.get("true_anomaly", preds)

# Classification report and confusion matrix
report = classification_report(true_labels, preds, output_dict=False)
cm = confusion_matrix(true_labels, preds)

# Save classification report
with open("metrics.txt", "w") as f:
    f.write("Isolation Forest Anomaly Detection Report\n")
    f.write("=" * 50 + "\n")
    f.write(report)
print("📄 Saved metrics.txt")

# Save confusion matrix plot
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("🖼️ Saved confusion_matrix.png")

# ROC curve and AUC
fpr, tpr, thresholds = roc_curve(true_labels, scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_auc_curve.png")
plt.close()
print("🖼️ Saved roc_auc_curve.png")

# Save model and scaler
joblib.dump(model, "anomaly_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model and scaler saved successfully.")
