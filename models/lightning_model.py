import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


# Define data preprocessing
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((47, 57)),  # Resize to 47x57
        transforms.ToTensor(),  # Convert to tensor
    ]
)

# Load training, validation, and test data
data_dir = "./data/train/train"
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)

validation_data_dir = "./data/validation/validation"
validation_dataset = datasets.ImageFolder(root=validation_data_dir, transform=transform)
validation_loader = DataLoader(validation_dataset, batch_size=30, shuffle=False)

test_data_dir = "./data/test/test"
test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)


class LitCNNModel(pl.LightningModule):
    def __init__(self, num_classes=22, learning_rate=1.0):
        super(LitCNNModel, self).__init__()
        self.save_hyperparameters()  # Save hyperparameters for logging and checkpointing
        self.conv1 = nn.Conv2d(1, 128, kernel_size=5)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # Use dummy input to calculate flattened size dynamically
        dummy_input = torch.zeros(1, 1, 47, 57)  # Batch size = 1, Grayscale channels, 47x57 image
        out = self._forward_conv(dummy_input)
        self.flattened_size = out.numel()

        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

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

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adadelta(self.parameters(), lr=self.hparams.learning_rate)


def calculate_model_accuracy(model, dataloader, device):
    model.eval()
    model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == "__main__":
    csv_logger = CSVLogger("logs", name="cnn_model")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1, dirpath="checkpoints", filename="best_model"
    )

    model = LitCNNModel(num_classes=len(train_dataset.classes))
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",  # Automatically selects GPU if available
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
        logger=csv_logger,
    )

    trainer.fit(model, train_loader, validation_loader)

    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    best_model = LitCNNModel.load_from_checkpoint(best_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calculate_model_accuracy(best_model, test_loader, device)
