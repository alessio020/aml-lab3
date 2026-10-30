import os
import torch
from torch import nn
import wandb

from models.customnet import CustomNet
from dataset.tiny_imagenet import get_dataloaders
from eval import validate


def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()              # 1. Azzera la memoria dei vecchi errori
        outputs = model(inputs)            # 2. Calcola le previsioni
        loss = criterion(outputs, targets) # 3. Calcola quanto ha sbagliato
        loss.backward()                    # 4. Calcola i gradienti (di chi è la colpa)
        optimizer.step()                   # 5. Aggiorna i pesi (impara dai propri errori)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')
    return train_loss, train_accuracy


if __name__ == '__main__':
    # Hyperparameters
    lr = 0.001
    momentum = 0.9
    batch_size = 32
    num_epochs = 10

    # 1. Start a W&B run
    wandb.init(
        project="tiny-imagenet-lab3",
        config={
            "learning_rate": lr,
            "momentum": momentum,
            "batch_size": batch_size,
            "epochs": num_epochs,
            "architecture": "CustomNet",
        }
    )

    # 2. Prepare DataLoaders, Model, Loss and Optimizer
    train_loader, val_loader, _, _ = get_dataloaders(batch_size=batch_size)
    model = CustomNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    best_acc = 0.0
    os.makedirs('checkpoints', exist_ok=True)

    # 3. Run the training process for {num_epochs} epochs
    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train(epoch, model, train_loader, criterion, optimizer)

        # At the end of each training iteration, perform a validation step
        val_accuracy = validate(model, val_loader, criterion)

        # Log metrics over time to visualize performance
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
        })

        # Best validation accuracy and model checkpoint saving
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')

    print(f'Best validation accuracy: {best_acc:.2f}%')
    wandb.finish()

