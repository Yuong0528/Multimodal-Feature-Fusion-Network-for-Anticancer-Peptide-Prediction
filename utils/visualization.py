import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc


def plot_confusion_matrix(y_true, y_prob, save_path):
    y_pred = (np.array(y_prob) >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_mean_roc(all_y_trues, all_y_probs, save_path):
    """
    sGROCò¿Øþ
    all_y_trues: List[List[int]]
    all_y_probs: List[List[float]]
    """
    from scipy import interp

    fpr_list = []
    tpr_list = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for y_true, y_prob in zip(all_y_trues, all_y_probs):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tpr_list.append(interp_tpr)
        aucs.append(auc(fpr, tpr))

    mean_tpr = np.mean(tpr_list, axis=0)
    std_tpr = np.std(tpr_list, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    plt.figure()
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=f'Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})', lw=2)
    plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='blue', alpha=0.2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Mean ROC Curve (5-Fold)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_bar_metrics(metric_dict, save_path):
    """
    Ø6ñ¶þ&G< ± Æî

    metric_dict: dict
        eg: {"acc": [0.81, 0.83, 0.85, 0.84, 0.82], "f1": [...], "auc": [...]}
    """
    metrics = list(metric_dict.keys())
    means = [np.mean(metric_dict[m]) for m in metrics]
    stds = [np.std(metric_dict[m]) for m in metrics]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, means, yerr=stds, capsize=6, color='skyblue', edgecolor='black')

    # èp<
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.01,
                 f"{mean:.4f}±{std:.4f}", ha='center', va='bottom', fontsize=9)

    plt.ylim(0, 1.1)
    plt.ylabel("Score")
    plt.title("5-Fold Cross-Validation Metrics")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
