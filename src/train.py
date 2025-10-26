import torch
from tqdm import trange
from sklearn.metrics import f1_score

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=10):
    model.to(device)
    for epoch in trange(num_epochs, desc="Training Progress"):
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss, running_f1 = 0.0, 0.0

            for inputs, targets in dataloaders[phase]:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = torch.sigmoid(model(inputs))
                    loss = criterion(outputs, targets)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                preds = (outputs > 0.5).float()
                running_loss += loss.item() * inputs.size(0)
                running_f1 += f1_score(
                    targets.cpu(), preds.cpu(), average="macro", zero_division=0
                ) * inputs.size(0)

            scheduler.step()
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_f1 = running_f1 / len(dataloaders[phase].dataset)
            print(f"[{phase}] Loss: {epoch_loss:.4f} | F1_macro: {epoch_f1:.4f}")
    return model
