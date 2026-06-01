import matplotlib.pyplot as plt

def plot_curves(losses, accuracies):
    plt.figure (figsize=(12,5))

    # Loss
    plt.subplot(1,2,1)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(accuracies)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.show()