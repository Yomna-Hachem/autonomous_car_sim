import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import os
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load your data CSV file
data_csv = 'new_driving_log.csv' 


# Load the data CSV into a dataframe
data_df = pd.read_csv(data_csv)

# Split the data into training and testing sets (70% training, 30% testing)
train_data, test_data = train_test_split(data_df, test_size=0.3, random_state=42)


# data set class
class Data(Dataset):
    def __init__(self, data_df, root_dir, transform=None):
        self.data = data_df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name)
        label = torch.tensor(self.data.iloc[idx, 1:3].astype('float32').values, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

# Data transformations
transform = transforms.Compose([
    transforms.Resize((75, 75)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Adjust mean and std values
])

# Create training and testing datasets and dataloaders
train_dataset = Data(data_df=train_data, root_dir='images', transform=transform)
test_dataset = Data(data_df=test_data, root_dir='images', transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=30, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=30, shuffle=False)


#my neural network

class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
    
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        print(self.maxpool1)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)


        self.fc1 = nn.Linear(784, 128)  
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.15)

        self.fc2 = nn.Linear(128, 64)
        self.relu5 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.15)

        self.fc_output = nn.Linear(64, 2)

    def forward(self, X):
        X = self.maxpool1(self.relu1(self.conv1(X)))
        X = self.maxpool2(self.relu2(self.conv2(X)))
        X = self.maxpool3(self.relu3(self.conv3(X)))

        # Flatten the output of the last convolutional layer
        X = X.view(X.size(0), -1)

        X = self.relu4(self.fc1(X))
        X = self.dropout1(X)

        X = self.relu5(self.fc2(X))
        X = self.dropout2(X)

        # Output layer for steering and throttle
        output = self.fc_output(X)

        return output


        
model = ConvNN().to(device)
epochs = 15
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0012)

# Training loop
for epoch in range(epochs):
    model.train()
    for images, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for images, labels in test_dataloader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

        average_loss = total_loss / len(test_dataloader)
        print(f'Epoch [{epoch + 1}/{epochs}] Validation Loss: {average_loss:.4f}')
        

