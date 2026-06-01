import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sklearn.model_selection import train_test_split

def train_model(
        model,
        X,
        y,
        device
):
    #split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size= 0.2, random_state= 40, stratify=y
    )

    #tensor datasets

    train_dataset = TensorDataset(
        X_train,
        y_train
    )

    val_dataset = TensorDataset(
        X_val, y_val
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32
    )

    #loss+optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr = 0.001
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode = 'min',
        patience=3
    )

    #Training

    epochs = 20
    train_losses = []
    val_accuracies = []
    best_accuracy = 0

    for epoch in range(epochs):
        #Train
        model.train()
        running_loss = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(
                outputs, labels
            )
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss =(running_loss / len(train_loader))

        train_losses.append(avg_loss)

        #Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                predictions = torch.argmax(
                    outputs, dim=1
                ) 

                correct += (
                    predictions == labels
                ).sum().item()

                total += labels.size(0)
        accuracy = correct / total
        val_accuracies.append(accuracy)
        scheduler.step(avg_loss)
        print(
            f"Epoch {epoch+1} | "
            f"Loss: {avg_loss:.4f} |"
            f"Val Accuracy: {accuracy:.4f}"
        )

        #Save Best Model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            os.makedirs("save_models", exist_ok=True)

            torch.save(
                model.state_dict(),
                "saved_models/best_model.pth"
            )
            print("Best Model Saved")
    return train_losses, val_accuracies