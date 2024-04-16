import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm

TRAIN = "/Users/supradparashar/Suprad/myCode/ML Datasets/MNIST Sign Language/sign_mnist_train/sign_mnist_train.csv"
TEST = "/Users/supradparashar/Suprad/myCode/ML Datasets/MNIST Sign Language/sign_mnist_test/sign_mnist_test.csv"



class SignLanguageDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.y = self.data['label'].values
        self.X = np.float32(self.data.drop('label', axis=1).values / 255.0)
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx].reshape(1, 28, 28), self.y[idx]
    
class SignLanguageModel(nn.Module):
    def __init__(self,channels=None, dropout_rate=0.5, kernel_size=3, stride=1, padding=1):
        super(SignLanguageModel, self).__init__()
        if channels is None:
            self.channels = [1, 4, 16]
        self.channels = channels
        self.dropout_rate = dropout_rate

        self.cnns = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            self.cnns.append(self.get_cnn_layer(self.channels[i], self.channels[i+1], kernel_size, stride, padding))
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * self.channels[-1], 64),  
            nn.Dropout(dropout_rate),
            nn.Linear(64, 26) 
        )
        
    
    def get_cnn_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_rate)
        )
        
    def forward(self, x):
        for cnn in self.cnns:
            x = cnn(x)
        x = x.reshape(x.shape[0], -1) # flatten the output of the last CNN layer
        x = self.fc(x)
        return x
    
if __name__ == "__main__":
    train_dataloader = DataLoader(SignLanguageDataset(TRAIN), batch_size=32, shuffle=True)
    test_dataloader = DataLoader(SignLanguageDataset(TEST), batch_size=32, shuffle=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = SignLanguageModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)

    print("Training the model...")

    epochs = 15
    for epoch in range(epochs):
        model.train()
        for X, y in tqdm(train_dataloader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item()}")

    print("Testing the model...")
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for X, y in tqdm(test_dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        print(f"Accuracy: {correct/total}")

    torch.save(model.state_dict(), "sign_language_model.pth")