import joblib
import numpy as np
import matplotlib.pyplot as plt

log_accuracy = 0.6072769020179273
log_title = "Logistic Regression with 80 organization topics, 15% test size, saga solver"

mlp_accuracy = 0.6151606392482546
mlp_title = "MLP Classifier with hidden layers (512,256,128)"

models = ['Logistic Regression', 'MLP Classifier']
titles = [log_title, mlp_title]

accuracies = [log_accuracy, mlp_accuracy]

plt.bar(models, accuracies, color=['blue', 'orange'], width=0.3)
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.yticks(np.arange(0, 1.1, 0.1))
plt.title('Comparison of accuracies')
plt.tight_layout()

for i, detail in enumerate(titles):
    plt.text(i, -0.06, detail, ha='center', va='top', fontsize=8, wrap=True, color='dimgray')


for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.02, f"{acc:.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.show()