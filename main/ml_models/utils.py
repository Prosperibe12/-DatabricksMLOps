import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import mlflow

def log_confusion_matrix(y_test, y_pred, labels, artifact_path):
    # compute confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    # plot figure
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title('Confusion Matrix')
    # log figure
    mlflow.log_figure(figure=fig, artifact_file=artifact_path)