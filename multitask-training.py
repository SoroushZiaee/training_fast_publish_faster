import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np

class MultiTaskImageDataset(Dataset):
    def __init__(self, num_samples=1000, image_size=(224, 224)):
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = Image.fromarray(np.random.randint(0, 255, (self.image_size[0], self.image_size[1], 3), dtype=np.uint8))
        image = self.transform(image)
        
        class_label = torch.randint(0, 3, (1,)).item()
        reg_label = torch.randn(1, dtype=torch.float32).item()  # Ensure float32
        
        return image, class_label, reg_label

class MultiTaskAlexNet(nn.Module):
    def __init__(self, num_classes):
        super(MultiTaskAlexNet, self).__init__()
        alexnet = models.alexnet(pretrained=False)
        
        for param in alexnet.parameters():
            param.requires_grad = False
        
        self.features = alexnet.features
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True)
        )
        
        self.classification_layer = nn.Linear(1024, num_classes)
        self.regression_layer = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        class_output = self.classification_layer(x)
        reg_output = self.regression_layer(x)
        return class_output, reg_output

def train(model, train_loader, optimizer, classification_criterion, regression_criterion, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, target_class, target_reg) in enumerate(train_loader):
        data, target_class, target_reg = data.to(device), target_class.to(device), target_reg.to(device)
        optimizer.zero_grad()
        class_output, reg_output = model(data)
        loss_class = classification_criterion(class_output, target_class)
        loss_reg = regression_criterion(reg_output, target_reg.unsqueeze(1).float())  # Ensure float32
        loss = loss_class + loss_reg
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type(torch.FloatTensor)  # Ensure default tensor type is float32
    
    dataset = MultiTaskImageDataset()
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MultiTaskAlexNet(num_classes=3).to(device)

    classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        loss = train(model, train_loader, optimizer, classification_criterion, regression_criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

if __name__ == "__main__":
    main()