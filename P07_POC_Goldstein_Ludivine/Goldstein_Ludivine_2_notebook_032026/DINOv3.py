# DINOv3 ViT-B/16 – Classification de 120 races de chiens
# Modèle : facebook/dinov3-vitb16-pretrain-lvd1689m (Meta AI)
# Dépendances : torch, torchvision, transformers>=4.56.0, scikit-learn, pandas, matplotlib, pillow
# Pour utiliser le modèle (gated), accepter la licence et créer un token HF.

import os
import pickle
import time
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoImageProcessor, AutoModel

print("=" * 70)
print("DINOv3 ViT-B/16 - ENTRAÎNEMENT SUR 120 RACES DE CHIENS".center(70))
print("=" * 70)

# ============================================================================
# CONFIGURATION GPU / CPU
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nAppareil utilisé : {device}")
if device.type == "cuda":
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM disponible : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# HYPERPARAMÈTRES
# ============================================================================

MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS_P1   = 20    # Phase 1 : tête seule
EPOCHS_P2   = 50    # Phase 2 : fine-tuning (avec early stopping)
LR_P1       = 5e-4
LR_P2       = 1e-4
WEIGHT_DECAY = 0.01
PATIENCE    = 15    # Early stopping
NUM_WORKERS = 4

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================

path_base = "Images"

def load_all_dogs(base_path):
    data_list = []
    all_folders = [f for f in os.listdir(base_path)
                   if os.path.isdir(os.path.join(base_path, f))]
    print(f"Nombre de races trouvées : {len(all_folders)}\n")
    total = 0
    for folder in all_folders:
        full_path = os.path.join(base_path, folder)
        images = glob(os.path.join(full_path, "*.jpg"))
        breed_name = folder.split('-')[-1]
        total += len(images)
        for img in images:
            data_list.append({"image_path": img, "label_name": breed_name})
    print(f"Total images chargées : {total}\n")
    return pd.DataFrame(data_list)

print("Chargement des images...")
data_full = load_all_dogs(path_base)

# ============================================================================
# 2. ENCODING DES LABELS
# ============================================================================

le = LabelEncoder()
data_full['label'] = le.fit_transform(data_full['label_name'])
num_classes = data_full['label'].nunique()

print(f"Nombre de classes : {num_classes}")
print(f"Total images      : {len(data_full)}")
print("\nDistribution (10 races les plus représentées) :")
print(data_full['label_name'].value_counts().head(10))

# ============================================================================
# 3. SPLIT TRAIN / VAL / TEST (70% / 15% / 15%)
# ============================================================================

train_df, temp_df = train_test_split(
    data_full, test_size=0.3, stratify=data_full['label'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
)

print(f"\n{'='*70}")
print(f"{'RÉPARTITION DES DONNÉES':^70}")
print(f"{'='*70}")
print(f"Train : {len(train_df):5d} images ({len(train_df)/len(data_full)*100:.1f}%)")
print(f"Val   : {len(val_df):5d} images ({len(val_df)/len(data_full)*100:.1f}%)")
print(f"Test  : {len(test_df):5d} images ({len(test_df)/len(data_full)*100:.1f}%)")
print(f"{'='*70}\n")

# ============================================================================
# 4. CHARGEMENT DU PROCESSEUR D'IMAGE DINOv3
# ============================================================================

print(f"Chargement du processeur DINOv3 depuis HuggingFace...")
print(f"  Modèle : {MODEL_NAME}\n")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

# Récupérer les paramètres de normalisation du processeur DINOv3
norm_mean = processor.image_mean if hasattr(processor, 'image_mean') else [0.485, 0.456, 0.406]
norm_std  = processor.image_std  if hasattr(processor, 'image_std')  else [0.229, 0.224, 0.225]

# ============================================================================
# 5. DATASET PYTORCH
# ============================================================================

class DogBreedDataset(Dataset):
    def __init__(self, df, transform=None):
        self.paths   = df['image_path'].values
        self.labels  = df['label'].values
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# Transformations avec augmentation pour l'entraînement
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_mean, std=norm_std),
])

# Sans augmentation pour val/test
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_mean, std=norm_std),
])

train_dataset = DogBreedDataset(train_df, transform=train_transform)
val_dataset   = DogBreedDataset(val_df,   transform=val_transform)
test_dataset  = DogBreedDataset(test_df,  transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

print(f"{'='*70}")
print(f"{'DATALOADERS CRÉÉS':^70}")
print(f"{'='*70}")
print(f"Train : {len(train_loader):4d} batches  ({len(train_dataset)} images)")
print(f"Val   : {len(val_loader):4d} batches  ({len(val_dataset)} images)")
print(f"Test  : {len(test_loader):4d} batches  ({len(test_dataset)} images)")
print(f"{'='*70}\n")

# ============================================================================
# 6. CONSTRUCTION DU MODÈLE DINOv3 + TÊTE DE CLASSIFICATION
# ============================================================================

class DINOv3Classifier(nn.Module):
    """DINOv3 backbone + tête de classification fine-tunable."""

    def __init__(self, backbone, hidden_size: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        # pooler_output = représentation CLS token (batch, hidden_size)
        features = outputs.pooler_output
        return self.classifier(features)

print(f"Chargement du backbone DINOv3 ViT-B/16...")
backbone = AutoModel.from_pretrained(MODEL_NAME)

# Taille du vecteur de sortie du backbone ViT-B
hidden_size = backbone.config.hidden_size  # 768 pour ViT-B
print(f"  Dimension du vecteur de features : {hidden_size}")

model = DINOv3Classifier(backbone, hidden_size, num_classes).to(device)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n{'='*70}")
print(f"{'DINOv3 ViT-B/16 - ARCHITECTURE':^70}")
print(f"{'='*70}")
print(f"Modèle          : DINOv3 ViT-B/16 (Meta AI, août 2025)")
print(f"Paramètres total: {total_params:,}")
print(f"Input size      : {IMG_SIZE}x{IMG_SIZE}")
print(f"Nombre classes  : {num_classes}")
print(f"{'='*70}\n")

# ============================================================================
# 7. FONCTIONS D'ENTRAÎNEMENT ET D'ÉVALUATION
# ============================================================================

criterion = nn.CrossEntropyLoss()

def train_one_epoch(model, loader, optimizer, scheduler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, correct, top5_correct, total = 0.0, 0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()

        # Top-5
        top5_preds = logits.topk(5, dim=1).indices
        top5_correct += (top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total, top5_correct / total

# ============================================================================
# 8. PHASE 1 : ENTRAÎNEMENT DE LA TÊTE SEULE (backbone gelé)
# ============================================================================

print("=" * 70)
print("PHASE 1 : ENTRAÎNEMENT DE LA TÊTE (backbone gelé)".center(70))
print("=" * 70 + "\n")

# Geler tout le backbone
for param in model.backbone.parameters():
    param.requires_grad = False

trainable_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Paramètres entraînables : {trainable_p1:,} (tête seule)\n")

optimizer_p1 = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_P1, weight_decay=WEIGHT_DECAY
)

history = {k: [] for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_top5']}
best_val_acc = 0.0

for epoch in range(1, EPOCHS_P1 + 1):
    t0 = time.time()
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer_p1)
    vl_loss, vl_acc, vl_top5 = evaluate(model, val_loader)
    elapsed = time.time() - t0

    history['train_loss'].append(tr_loss)
    history['train_acc'].append(tr_acc)
    history['val_loss'].append(vl_loss)
    history['val_acc'].append(vl_acc)
    history['val_top5'].append(vl_top5)

    marker = " *" if vl_acc > best_val_acc else ""
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save(model.state_dict(), 'best_dinov3_120races.pt')

    print(f"Epoch {epoch:3d}/{EPOCHS_P1} | "
          f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} | "
          f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}  Top5: {vl_top5:.4f} | "
          f"{elapsed:.0f}s{marker}")

phase1_epochs = len(history['train_loss'])

# ============================================================================
# 9. PHASE 2 : FINE-TUNING (dégel des dernières couches du backbone)
# ============================================================================

print("\n" + "=" * 70)
print("PHASE 2 : FINE-TUNING (dégel partiel du backbone)".center(70))
print("=" * 70 + "\n")

# Dégeler tout le backbone
for param in model.backbone.parameters():
    param.requires_grad = True

# Regeler les N premières couches du transformer (encoder)
# Pour ViT-B : 12 blocs au total, on gèle les 6 premiers
backbone_layers = list(model.backbone.encoder.layer) \
    if hasattr(model.backbone, 'encoder') and hasattr(model.backbone.encoder, 'layer') \
    else []

freeze_until = max(0, len(backbone_layers) - 6)  # garder les 6 derniers blocs dégelés
for layer in backbone_layers[:freeze_until]:
    for param in layer.parameters():
        param.requires_grad = False

trainable_p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Blocs totaux dans le backbone : {len(backbone_layers)}")
print(f"Blocs dégelés                 : {len(backbone_layers) - freeze_until}")
print(f"Paramètres entraînables       : {trainable_p2:,}\n")

optimizer_p2 = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_P2, weight_decay=0.05
)

scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_p2, T_max=EPOCHS_P2, eta_min=1e-7
)

patience_counter = 0

for epoch in range(1, EPOCHS_P2 + 1):
    t0 = time.time()
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer_p2)
    vl_loss, vl_acc, vl_top5 = evaluate(model, val_loader)
    scheduler_p2.step()
    elapsed = time.time() - t0

    history['train_loss'].append(tr_loss)
    history['train_acc'].append(tr_acc)
    history['val_loss'].append(vl_loss)
    history['val_acc'].append(vl_acc)
    history['val_top5'].append(vl_top5)

    marker = ""
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        patience_counter = 0
        torch.save(model.state_dict(), 'best_dinov3_120races.pt')
        marker = " *"
    else:
        patience_counter += 1

    total_epoch = phase1_epochs + epoch
    print(f"Epoch {total_epoch:3d} (FT {epoch:2d}/{EPOCHS_P2}) | "
          f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} | "
          f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}  Top5: {vl_top5:.4f} | "
          f"LR: {scheduler_p2.get_last_lr()[0]:.2e} | "
          f"{elapsed:.0f}s{marker}")

    if patience_counter >= PATIENCE:
        print(f"\n[Early Stopping] Pas d'amélioration depuis {PATIENCE} epochs. Arrêt.")
        break

# ============================================================================
# 10. ÉVALUATION FINALE (avec le meilleur modèle sauvegardé)
# ============================================================================

print("\n" + "=" * 70)
print("ÉVALUATION FINALE".center(70))
print("=" * 70 + "\n")

model.load_state_dict(torch.load('best_dinov3_120races.pt', map_location=device))
model.eval()

print("Évaluation sur TRAIN...")
train_loss, train_acc, train_top5 = evaluate(model, train_loader)

print("Évaluation sur VAL...")
val_loss, val_acc, val_top5 = evaluate(model, val_loader)

print("Évaluation sur TEST...")
test_loss, test_acc, test_top5 = evaluate(model, test_loader)

# ============================================================================
# 11. AFFICHAGE DES RÉSULTATS
# ============================================================================

print(f"\n{'='*70}")
print(f"{'DINOv3 ViT-B/16 - RÉSULTATS FINAUX':^70}")
print(f"{'='*70}")
print(f"{'Dataset':<12} {'Loss':<12} {'Accuracy':<12} {'Top-5 Acc':<12}")
print(f"{'-'*70}")
print(f"{'Train':<12} {train_loss:<12.4f} {train_acc:<12.4f} {train_top5:<12.4f}")
print(f"{'Val':<12}   {val_loss:<12.4f} {val_acc:<12.4f} {val_top5:<12.4f}")
print(f"{'Test':<12}  {test_loss:<12.4f} {test_acc:<12.4f} {test_top5:<12.4f}")
print(f"{'='*70}")

overfit_tv = (train_acc - val_acc) * 100
overfit_vt = (val_acc - test_acc) * 100
print(f"\nÉcart Train-Val  : {overfit_tv:+.2f}%")
print(f"Écart Val-Test   : {overfit_vt:+.2f}%")

if overfit_tv > 10:
    print("Overfitting détecté (Train >> Val)")
if abs(overfit_vt) > 5:
    print("Écart significatif Val-Test")

# ============================================================================
# 12. COURBES D'APPRENTISSAGE
# ============================================================================

epochs_range = range(1, len(history['train_acc']) + 1)
phase1_end = phase1_epochs

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("DINOv3 ViT-B/16 — Entraînement 120 races de chiens", fontsize=15, fontweight='bold')

# Accuracy
axes[0, 0].plot(epochs_range, history['train_acc'], label='Train', linewidth=2, color='#1f77b4')
axes[0, 0].plot(epochs_range, history['val_acc'],   label='Val',   linewidth=2, color='#ff7f0e')
axes[0, 0].axhline(y=test_acc, color='r', linestyle='--',
                   label=f'Test ({test_acc:.3f})', linewidth=2)
axes[0, 0].axvline(x=phase1_end, color='gray', linestyle=':', alpha=0.6, label='Fine-tuning')
axes[0, 0].set_title('Accuracy par epoch', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(epochs_range, history['train_loss'], label='Train', linewidth=2, color='#1f77b4')
axes[0, 1].plot(epochs_range, history['val_loss'],   label='Val',   linewidth=2, color='#ff7f0e')
axes[0, 1].axhline(y=test_loss, color='r', linestyle='--',
                   label=f'Test ({test_loss:.3f})', linewidth=2)
axes[0, 1].axvline(x=phase1_end, color='gray', linestyle=':', alpha=0.6, label='Fine-tuning')
axes[0, 1].set_title('Loss par epoch', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Top-5 Accuracy
axes[1, 0].plot(epochs_range, history['val_top5'], label='Val Top-5', linewidth=2, color='#2ca02c')
axes[1, 0].axhline(y=test_top5, color='r', linestyle='--',
                   label=f'Test Top-5 ({test_top5:.3f})', linewidth=2)
axes[1, 0].axvline(x=phase1_end, color='gray', linestyle=':', alpha=0.6, label='Fine-tuning')
axes[1, 0].set_title('Top-5 Accuracy (Val)', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Top-5 Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Comparaison finale (barres)
categories = ['Train', 'Val', 'Test']
accuracies = [train_acc, val_acc, test_acc]
top5_accs  = [train_top5, val_top5, test_top5]
x = np.arange(len(categories))
width = 0.35

bars1 = axes[1, 1].bar(x - width/2, accuracies, width, label='Top-1 Accuracy', alpha=0.85, color='#2ca02c')
bars2 = axes[1, 1].bar(x + width/2, top5_accs,  width, label='Top-5 Accuracy', alpha=0.85, color='#d62728')
axes[1, 1].set_title('Résultats finaux', fontsize=13, fontweight='bold')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(categories)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')
axes[1, 1].set_ylim([0, 1])

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('dinov3_training_results.png', dpi=300, bbox_inches='tight')
print(f"\nGraphiques sauvegardés : dinov3_training_results.png")
plt.show()

# ============================================================================
# 13. SAUVEGARDE DES RÉSULTATS ET ARTEFACTS
# ============================================================================

total_epochs_trained = len(history['train_loss'])
ft_epochs = total_epochs_trained - phase1_epochs

with open('dinov3_results.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("DINOv3 ViT-B/16 - RÉSULTATS FINAUX\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Modèle            : DINOv3 ViT-B/16 (Meta AI, août 2025)\n")
    f.write(f"HuggingFace ID    : {MODEL_NAME}\n")
    f.write(f"Nombre de classes : {num_classes}\n")
    f.write(f"Total images      : {len(data_full)}\n")
    f.write(f"Train             : {len(train_df)} images\n")
    f.write(f"Val               : {len(val_df)} images\n")
    f.write(f"Test              : {len(test_df)} images\n\n")
    f.write(f"Paramètres totaux         : {total_params:,}\n")
    f.write(f"Epochs entraînés          : {total_epochs_trained}\n")
    f.write(f"  - Phase 1 (head only)   : {phase1_epochs} epochs\n")
    f.write(f"  - Phase 2 (fine-tuning) : {ft_epochs} epochs\n\n")
    f.write(f"{'Dataset':<12} {'Loss':<12} {'Accuracy':<12} {'Top-5 Acc':<12}\n")
    f.write(f"{'-'*70}\n")
    f.write(f"{'Train':<12} {train_loss:<12.4f} {train_acc:<12.4f} {train_top5:<12.4f}\n")
    f.write(f"{'Val':<12} {val_loss:<12.4f} {val_acc:<12.4f} {val_top5:<12.4f}\n")
    f.write(f"{'Test':<12} {test_loss:<12.4f} {test_acc:<12.4f} {test_top5:<12.4f}\n")
    f.write("=" * 70 + "\n")

print("Résultats sauvegardés : dinov3_results.txt")

with open('label_encoder_dinov3.pkl', 'wb') as f:
    pickle.dump(le, f)
print("LabelEncoder sauvegardé : label_encoder_dinov3.pkl")

print("\n" + "=" * 70)
print("ENTRAÎNEMENT TERMINÉ !".center(70))
print("=" * 70)
print("\nFichiers générés :")
print("  - best_dinov3_120races.pt        (meilleur modèle PyTorch)")
print("  - dinov3_training_results.png    (graphiques)")
print("  - dinov3_results.txt             (résultats texte)")
print("  - label_encoder_dinov3.pkl       (pour prédictions)")
