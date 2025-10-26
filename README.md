# 🔥 Multi-Label Classification Model (ResNet50 + Focal Loss)

This repository contains a modular and professional implementation of a multi-label image classification model using **PyTorch**.

> ⚠️ **Note:** This repository includes only the *model and training code*.
> Image preprocessing and data pipeline definitions are **not included**.

---

## 🧠 Features
- Transfer learning with ResNet50 backbone
- Multi-label classification with sigmoid activation
- Focal Loss for handling label imbalance
- Modular structure (train, evaluate, model)
- Comprehensive metric evaluation (F1, mAP)

---

## 📦 Installation
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage
```bash
python main.py
```

---

## 📊 Example Output
```
[train] Loss: 0.3271 | F1_macro: 0.8423
[val]   Loss: 0.3559 | F1_macro: 0.8105
F1_macro: 0.8105, mAP: 0.7932
```

---

## 📜 License
MIT License
