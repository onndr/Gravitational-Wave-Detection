import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pickle

# Define data preprocessing
transform = transforms.Compose(
    [
        transforms.Resize((47, 57)),  # Resize to 47x57
        transforms.ToTensor(),  # Convert to tensor (preserves RGB channels)
    ]
)

# Load training data
data_dir = "./data/train/train"
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)


# Define the CNN architecture
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=5)  # Input channels for RGB
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=5)
        self.fc1 = nn.Linear(128 * 9 * 12, 256)  # Flatten size after Conv+Pool
        self.fc2 = nn.Linear(256, 20)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 128 * 9 * 12)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)


# Training function
def train_model(model, train_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss
    optimizer = optim.Adadelta(model.parameters())  # Adadelta optimizer

    model.train()  # Set the model to training mode
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

    print("Training complete!")


# Save the model as a pickle file
def save_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")


# Main execution
if __name__ == "__main__":
    # Initialize the model
    model = CNNModel()

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model
    train_model(model, train_loader, num_epochs=130, device=device)

    # Save the trained model
    save_model(model, "cnn_model.pkl")
