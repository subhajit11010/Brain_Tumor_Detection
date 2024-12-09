import torch.nn as nn
import torch
import torch.optim as optim
def train(model, learning_rate, epochs, batch_size, device, train_loader, val_loader):
    optimizer  = optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs,labels in train_loader:
            inputs,labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs,1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            avg_val_loss = val_loss / len(val_loader)
            accuracy = 100 * correct / total
            print(f"Validation Loss: {avg_val_loss}, Accuracy: {accuracy}%")
