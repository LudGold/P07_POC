
# Modèle ConvNeXt - CODE COMPLET POUR 120 RACES DE CHIENS

import os
from glob import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

print("="*70)
print("ConvNeXt - ENTRAÎNEMENT SUR 120 RACES DE CHIENS".center(70))
print("="*70)

print(f"\nTensorFlow version: {tf.__version__}")

# Vérifier que TensorFlow >= 2.11 (requis pour ConvNeXt)
tf_version = tuple(map(int, tf.__version__.split('.')[:2]))
if tf_version < (2, 11):
    print(f"\n ERREUR : ConvNeXt nécessite TensorFlow >= 2.11")
    print(f"   Version actuelle : {tf.__version__}")
    print("\nSOLUTION : Mettez à jour TensorFlow :")
    print("  pip install --upgrade tensorflow")
    raise SystemExit(1)

print(f"GPU disponible: {tf.config.list_physical_devices('GPU')}\n")

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================

path_base = "Images"

def load_all_dogs(base_path):
    """Charge toutes les races de chiens du dataset"""
    data_list = []
    
    all_folders = [f for f in os.listdir(base_path) 
                   if os.path.isdir(os.path.join(base_path, f))]
    
    print(f"Nombre de races trouvées : {len(all_folders)}\n")
    
    total_images = 0
    for folder in all_folders:
        full_path = os.path.join(base_path, folder)
        images = glob(os.path.join(full_path, "*.jpg"))
        breed_name = folder.split('-')[-1]
        
        total_images += len(images)
        
        for img in images:
            data_list.append({"image_path": img, "label_name": breed_name})
    
    print(f"Total images chargées : {total_images}\n")
    
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
print(f"Total images : {len(data_full)}")

# Afficher distribution des 10 races les plus représentées
print("\nDistribution (10 races les plus représentées) :")
print(data_full['label_name'].value_counts().head(10))

# ============================================================================
# 3. SPLIT TRAIN / VAL / TEST (70% / 15% / 15%)
# ============================================================================

# Train vs (Val+Test)
train_df, temp_df = train_test_split(
    data_full, 
    test_size=0.3,
    stratify=data_full['label'], 
    random_state=42
)

# Val vs Test
val_df, test_df = train_test_split(
    temp_df, 
    test_size=0.5,
    stratify=temp_df['label'], 
    random_state=42
)

print(f"\n{'='*70}")
print(f"{'RÉPARTITION DES DONNÉES':^70}")
print(f"{'='*70}")
print(f"Train : {len(train_df):5d} images ({len(train_df)/len(data_full)*100:.1f}%)")
print(f"Val   : {len(val_df):5d} images ({len(val_df)/len(data_full)*100:.1f}%)")
print(f"Test  : {len(test_df):5d} images ({len(test_df)/len(data_full)*100:.1f}%)")
print(f"Total : {len(data_full):5d} images")
print(f"{'='*70}\n")

# ============================================================================
# 4. DATA AUGMENTATION POUR ConvNeXt
# ============================================================================

IMG_SIZE = 224
BATCH_SIZE = 32  # ConvNeXt est efficace, on peut mettre 32

print("Configuration du preprocessing pour ConvNeXt...")

# IMPORTANT : Utiliser le preprocessing spécifique de ConvNeXt
from tensorflow.keras.applications.convnext import preprocess_input

# Data augmentation pour ConvNeXt avec le BON preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Preprocessing ConvNeXt (ImageNet normalization)
    rotation_range=20,          # Rotation ±20° (réduit)
    width_shift_range=0.2,      # Décalage horizontal (réduit)
    height_shift_range=0.2,     # Décalage vertical (réduit)
    shear_range=0.15,           # Déformation (réduit)
    zoom_range=0.15,            # Zoom (réduit)
    horizontal_flip=True,       # Flip horizontal
    fill_mode='nearest'
)


# Pas d'augmentation pour VAL et TEST (juste le preprocessing ConvNeXt)
val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# ============================================================================
# 5. GÉNÉRATEURS DE DONNÉES
# ============================================================================

print("Création des générateurs de données...")

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='image_path',
    y_col='label_name',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

val_generator = val_test_datagen.flow_from_dataframe(
    val_df,
    x_col='image_path',
    y_col='label_name',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_dataframe(
    test_df,
    x_col='image_path',
    y_col='label_name',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"\n{'='*70}")
print(f"{'GÉNÉRATEURS CRÉÉS':^70}")
print(f"{'='*70}")
print(f"Train : {len(train_generator):4d} batches")
print(f"Val   : {len(val_generator):4d} batches")
print(f"Test  : {len(test_generator):4d} batches")
print(f"{'='*70}\n")

# ============================================================================
# 6. CONSTRUCTION DU MODÈLE ConvNeXt
# ============================================================================

print("Construction du modèle ConvNeXt-Tiny...\n")

# Charger la base ConvNeXt-Tiny pré-entraînée sur ImageNet
base_model = ConvNeXtTiny(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Geler les couches de base pour le début
base_model.trainable = False

# Ajouter les couches de classification
x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.5, name='dropout')(x)
x = Dense(512, activation='relu', name='fc1')(x)
x = Dropout(0.3, name='dropout2')(x)
predictions = Dense(num_classes, activation='softmax', name='predictions')(x)

model_convnext = Model(inputs=base_model.input, outputs=predictions, name='ConvNeXt_DogBreeds')

print(f"{'='*70}")
print(f"{'ConvNeXt-Tiny - ARCHITECTURE':^70}")
print(f"{'='*70}")
print(f"Modèle          : ConvNeXt-Tiny (Meta AI 2022)")
print(f"Paramètres      : {model_convnext.count_params():,}")
print(f"Paramètres train: {sum([tf.size(w).numpy() for w in model_convnext.trainable_weights]):,}")
print(f"Input size      : {IMG_SIZE}x{IMG_SIZE}")
print(f"Nombre classes  : {num_classes}")
print(f"Base trainable  : {base_model.trainable}")
print(f"{'='*70}\n")

# ============================================================================
# 7. COMPILATION DU MODÈLE
# ============================================================================

# Utiliser AdamW (meilleur pour les architectures type Transformer)
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=5e-4,  # LR initial réduit (0.0005)
    weight_decay=0.01    # Régularisation L2 réduite
)

model_convnext.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')
    ]
)

print("✓ Modèle compilé avec AdamW optimizer\n")

# ============================================================================
# 8. CALLBACKS POUR L'ENTRAÎNEMENT
# ============================================================================

callbacks = [
    # Early stopping : arrête si pas d'amélioration
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1,
        mode='max'
    ),
    
    # Réduction du learning rate
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    
    # Sauvegarde du meilleur modèle
    ModelCheckpoint(
        'best_convnext_120races.h5',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
]

# ============================================================================
# 9. ENTRAÎNEMENT PHASE 1 : Tête seule (base gelée)
# ============================================================================

print("\n" + "="*70)
print("PHASE 1 : ENTRAÎNEMENT DE LA TÊTE (base gelée)".center(70))
print("="*70 + "\n")

history_phase1 = model_convnext.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,  # Peu d'epochs pour la phase 1
    callbacks=callbacks,
    verbose=1
)

# ============================================================================
# 10. FINE-TUNING : Dégeler une partie de la base
# ============================================================================

print("\n" + "="*70)
print("PHASE 2 : FINE-TUNING (dégel partiel de la base)".center(70))
print("="*70 + "\n")

# Dégeler les dernières couches de la base
base_model.trainable = True

# Geler les premières couches, dégeler les dernières
for layer in base_model.layers[:-30]:  # Geler tout sauf les 30 dernières couches
    layer.trainable = False

print(f"Couches dégelées : {sum([1 for layer in base_model.layers if layer.trainable])}/{len(base_model.layers)}")
print(f"Paramètres entraînables : {sum([tf.size(w).numpy() for w in model_convnext.trainable_weights]):,}")

# Recompiler avec un learning rate plus faible pour le fine-tuning
optimizer_ft = tf.keras.optimizers.AdamW(
    learning_rate=1e-4,  # LR 10x plus faible
    weight_decay=0.05
)

model_convnext.compile(
    optimizer=optimizer_ft,
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')
    ]
)

print("✓ Modèle recompilé pour fine-tuning\n")

# Entraînement phase 2
history_phase2 = model_convnext.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,  # Plus d'epochs pour le fine-tuning
    callbacks=callbacks,
    verbose=1,
    initial_epoch=len(history_phase1.history['loss'])  # Continue depuis la phase 1
)

# Combiner les historiques
history = {
    'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
    'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
    'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
    'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss'],
    'top5_accuracy': history_phase1.history['top5_accuracy'] + history_phase2.history['top5_accuracy'],
    'val_top5_accuracy': history_phase1.history['val_top5_accuracy'] + history_phase2.history['val_top5_accuracy']
}

# ============================================================================
# 11. ÉVALUATION SUR LES 3 JEUX DE DONNÉES
# ============================================================================

print("\n" + "="*70)
print("ÉVALUATION FINALE".center(70))
print("="*70 + "\n")

print("Évaluation sur TRAIN...")
train_loss, train_acc, train_top5 = model_convnext.evaluate(train_generator, verbose=0)

print("Évaluation sur VAL...")
val_loss, val_acc, val_top5 = model_convnext.evaluate(val_generator, verbose=0)

print("Évaluation sur TEST...")
test_loss, test_acc, test_top5 = model_convnext.evaluate(test_generator, verbose=0)

# ============================================================================
# 12. AFFICHAGE DES RÉSULTATS
# ============================================================================

print(f"\n{'='*70}")
print(f"{'ConvNeXt-Tiny - RÉSULTATS FINAUX':^70}")
print(f"{'='*70}")
print(f"{'Dataset':<12} {'Loss':<12} {'Accuracy':<12} {'Top-5 Acc':<12}")
print(f"{'-'*70}")
print(f"{'Train':<12} {train_loss:<12.4f} {train_acc:<12.4f} {train_top5:<12.4f}")
print(f"{'Val':<12} {val_loss:<12.4f} {val_acc:<12.4f} {val_top5:<12.4f}")
print(f"{'Test':<12} {test_loss:<12.4f} {test_acc:<12.4f} {test_top5:<12.4f}")
print(f"{'='*70}")

# Analyse overfitting
overfit_train_val = (train_acc - val_acc) * 100
overfit_val_test = (val_acc - test_acc) * 100

print(f"\nÉcart Train-Val  : {overfit_train_val:+.2f}%")
print(f"Écart Val-Test   : {overfit_val_test:+.2f}%")

if overfit_train_val > 10:
    print("⚠️  Overfitting détecté (Train >> Val)")
elif overfit_train_val < -5:
    print("ℹ️  Underfitting possible (Val > Train)")

if abs(overfit_val_test) > 5:
    print("⚠️  Écart significatif Val-Test")

# ============================================================================
# 13. VISUALISATION DES COURBES D'APPRENTISSAGE
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

epochs_range = range(1, len(history['accuracy']) + 1)

# Accuracy
axes[0, 0].plot(epochs_range, history['accuracy'], label='Train Accuracy', linewidth=2, color='#1f77b4')
axes[0, 0].plot(epochs_range, history['val_accuracy'], label='Val Accuracy', linewidth=2, color='#ff7f0e')
axes[0, 0].axhline(y=test_acc, color='r', linestyle='--', 
                   label=f'Test Accuracy ({test_acc:.3f})', linewidth=2)
axes[0, 0].axvline(x=20, color='gray', linestyle=':', alpha=0.5, label='Fine-tuning start')
axes[0, 0].set_title('Accuracy par epoch', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(epochs_range, history['loss'], label='Train Loss', linewidth=2, color='#1f77b4')
axes[0, 1].plot(epochs_range, history['val_loss'], label='Val Loss', linewidth=2, color='#ff7f0e')
axes[0, 1].axhline(y=test_loss, color='r', linestyle='--', 
                   label=f'Test Loss ({test_loss:.3f})', linewidth=2)
axes[0, 1].axvline(x=20, color='gray', linestyle=':', alpha=0.5, label='Fine-tuning start')
axes[0, 1].set_title('Loss par epoch', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Top-5 Accuracy
axes[1, 0].plot(epochs_range, history['top5_accuracy'], label='Train Top-5', linewidth=2, color='#1f77b4')
axes[1, 0].plot(epochs_range, history['val_top5_accuracy'], label='Val Top-5', linewidth=2, color='#ff7f0e')
axes[1, 0].axhline(y=test_top5, color='r', linestyle='--', 
                   label=f'Test Top-5 ({test_top5:.3f})', linewidth=2)
axes[1, 0].axvline(x=20, color='gray', linestyle=':', alpha=0.5, label='Fine-tuning start')
axes[1, 0].set_title('Top-5 Accuracy par epoch', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Top-5 Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Comparaison finale
categories = ['Train', 'Val', 'Test']
accuracies = [train_acc, val_acc, test_acc]
top5_accs = [train_top5, val_top5, test_top5]

x = np.arange(len(categories))
width = 0.35

bars1 = axes[1, 1].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='#2ca02c')
bars2 = axes[1, 1].bar(x + width/2, top5_accs, width, label='Top-5 Accuracy', alpha=0.8, color='#d62728')

axes[1, 1].set_title('Résultats finaux', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(categories)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')
axes[1, 1].set_ylim([0, 1])

# Ajouter les valeurs sur les barres
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('convnext_training_results.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Graphiques sauvegardés : convnext_training_results.png")
plt.show()

# ============================================================================
# 14. SAUVEGARDE DES RÉSULTATS
# ============================================================================

with open('convnext_results.txt', 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("ConvNeXt-Tiny - RÉSULTATS FINAUX\n")
    f.write("="*70 + "\n\n")
    f.write(f"Modèle          : ConvNeXt-Tiny (Meta AI 2022)\n")
    f.write(f"Nombre de classes : {num_classes}\n")
    f.write(f"Total images : {len(data_full)}\n")
    f.write(f"Train : {len(train_df)} images\n")
    f.write(f"Val   : {len(val_df)} images\n")
    f.write(f"Test  : {len(test_df)} images\n\n")
    f.write(f"Paramètres du modèle : {model_convnext.count_params():,}\n")
    f.write(f"Epochs entraînés : {len(history['loss'])}\n")
    f.write(f"  - Phase 1 (head only) : 20 epochs\n")
    f.write(f"  - Phase 2 (fine-tuning) : {len(history['loss']) - 20} epochs\n\n")
    f.write(f"{'Dataset':<12} {'Loss':<12} {'Accuracy':<12} {'Top-5 Acc':<12}\n")
    f.write(f"{'-'*70}\n")
    f.write(f"{'Train':<12} {train_loss:<12.4f} {train_acc:<12.4f} {train_top5:<12.4f}\n")
    f.write(f"{'Val':<12} {val_loss:<12.4f} {val_acc:<12.4f} {val_top5:<12.4f}\n")
    f.write(f"{'Test':<12} {test_loss:<12.4f} {test_acc:<12.4f} {test_top5:<12.4f}\n")
    f.write("="*70 + "\n")

print("✓ Résultats sauvegardés : convnext_results.txt")

# Sauvegarder le LabelEncoder pour les prédictions futures
import pickle
with open('label_encoder_convnext.pkl', 'wb') as f:
    pickle.dump(le, f)
print("✓ LabelEncoder sauvegardé : label_encoder_convnext.pkl")

print("\n" + "="*70)
print("✓ ENTRAÎNEMENT TERMINÉ !".center(70))
print("="*70)
print("\nFichiers générés :")
print("  - best_convnext_120races.h5          (meilleur modèle)")
print("  - convnext_training_results.png      (graphiques)")
print("  - convnext_results.txt               (résultats texte)")
print("  - label_encoder_convnext.pkl         (pour prédictions)")
