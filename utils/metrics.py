# utils/metrics.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os


def compute_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    if save_path:
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    return cm
