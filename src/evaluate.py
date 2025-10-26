import torch
from sklearn.metrics import f1_score, precision_recall_curve, auc

def evaluate_model(model, dataloader, device):
    model.eval()
    preds_all, targets_all = [], []

    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = (torch.sigmoid(model(imgs)) > 0.5).float()
            preds_all.append(preds.cpu())
            targets_all.append(targets.cpu())

    preds_all = torch.cat(preds_all)
    targets_all = torch.cat(targets_all)

    f1_macro = f1_score(targets_all, preds_all, average="macro", zero_division=0)
    aps = []
    for i in range(targets_all.shape[1]):
        p, r, _ = precision_recall_curve(targets_all[:, i], preds_all[:, i])
        aps.append(auc(r, p))
    map_score = sum(aps) / len(aps)

    print(f"F1_macro: {f1_macro:.4f}, mAP: {map_score:.4f}")
    return f1_macro, map_score
