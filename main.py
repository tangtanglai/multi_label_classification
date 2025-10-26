import torch
from torch.utils.data import DataLoader, random_split
from torch import optim
from torch.optim import lr_scheduler
from src.model import MultiLabelResNet50
from src.train import train_model
from src.evaluate import evaluate_model
from focal_loss import FocalLoss
from dataset import MyDataset  # 注意：不包含图像预处理，仅模型逻辑

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_labels = ["Sägen", "Stanzen", "Entgraten", "Biegen/Präg.", "Bohren/Gew.",
                "Schweißen", "Punktschweiß.", "Schleifen", "Beschichten",
                "Laserschneider", "Revolverstanzmaschine"]

dataset = MyDataset("./paper/11/exp23.csv", "./jpg/", transform=None)
train_set, val_set = random_split(dataset, [len(dataset) - 12, 12])
dataloaders = {
    "train": DataLoader(train_set, batch_size=32, shuffle=True),
    "val": DataLoader(val_set, batch_size=32, shuffle=False)
}

model = MultiLabelResNet50(num_classes=len(class_labels))
criterion = FocalLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

model = train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=15)
torch.save(model.state_dict(), "./paper/log/exp23_refactored.pth")
evaluate_model(model, dataloaders["val"], device)
