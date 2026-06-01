import torch
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score
)
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_model(
        model, loader, device
):
    
    model.eval()
    all_predictions  = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            predictions = torch.argmax(
                outputs, dim =1
            )

            all_predictions.extend(
                predictions.cpu().numpy
            )

            all_labels.extend(
                labels.numpy()
            )

    # report
    print(classification_report(
        all_labels, all_predictions
    ))
    #F1
    f1 = f1_score(
        all_labels, all_predictions, average='weighted'
    )

    print("F1 Score:", f1)

    #confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    plt.figure(figsize=(6,5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()