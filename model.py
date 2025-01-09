import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pickle

# Define data preprocessing
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((47, 57)),  # Resize to 47x57
        transforms.ToTensor(),  # Convert to tensor
    ]
)


# Load training data
data_dir = "./data/train/train"
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)
# print(train_dataset.class_to_idx)
# print(len(train_dataset.classes))


# Define the CNN architecture
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=5)  # Input channels changed to 1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=5)
        self.relu = nn.ReLU()

        # Use dummy input to calculate flattened size dynamically
        dummy_input = torch.zeros(1, 1, 47, 57)  # Batch size = 1, Grayscale channels, 47x57 image
        out = self._forward_conv(dummy_input)
        self.flattened_size = out.numel()

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, 22)

    def _forward_conv(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten dynamically
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Output raw logits (no softmax here)
        return x


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
            outputs = model(inputs)  # Raw logits

            # Compute loss using raw logits and integer labels
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

    print("Training complete!")


# Save the model
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
