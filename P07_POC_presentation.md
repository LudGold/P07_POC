---
marp: true
title: P07 – Le Refuge : Identification de races de chiens
author: Ludivine Goldstein
theme: default
paginate: true
backgroundColor: #ffffff
---

<!-- _class: lead -->

# P07 – Le Refuge  
## Identification de races de chiens par Deep Learning

**Projet POC – Streamlit & Deep Learning**  
Dataset : Stanford Dogs (120 races, ~20 580 images)

---

## Contexte & objectifs

- **Contexte métier**
  - Un refuge canin doit identifier rapidement la **race** d’un chien à partir d’une **photo**.
  - L’identification manuelle est **lente**, **subjective** et nécessite une **expertise**.

- **Objectifs du POC**
  - Construire une **application web** simple d’usage (`Streamlit`).
  - Tester plusieurs **architectures de Deep Learning**.
  - Comparer les **performances** et la **vitesse d’inférence**.
  - Proposer une interface accessible aux **bénévoles du refuge**.

---

## Jeu de données : Stanford Dogs

- **Source**
  - [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)
  - 120 races, ~20 580 images

- **Préparation des données**
  - Split : **70 % train / 15 % validation / 15 % test**
  - Contrôle de la **distribution d’images par race**
  - Analyse des **dimensions d’images** (larg./haut.)

- **Challenge**
  - Forte **variabilité** (postures, luminosité, arrière‑plans)
  - Races parfois **visuellement proches**

---

## Analyse exploratoire (EDA) dans l’app

- Onglet **« Exploration des données »**
  - Statistiques globales :
    - Nombre total d’images
    - Nombre de races
    - Nombre moyen d’images par race
  - **Histogramme** : distribution du nombre d’images par race
  - **Boxplots** : largeur et hauteur des images

- Visualisation
  - Sélection d’une **race** et affichage de quelques **exemples d’images**
  - Mise en avant des **variations de qualité** et de taille

---

## Pré‑traitements & transformations

- Affichage d’une image et de trois transformations :
  - **Image originale** redimensionnée en \(224 \times 224\)
  - **Égalisation d’histogramme**
  - **Flou gaussien**
  - **Détection de contours**

- Intérêts :
  - Améliorer la **lisibilité** et le **contraste**
  - Explorer des **augmentations de données** possibles
  - Illustration pédagogique pour les **utilisateurs non techniques**

---

## Architectures de modèles testées

- **ConvNeXt-Tiny** (TensorFlow / Keras)
  - Backbone `ConvNeXtTiny` sans top
  - Global Average Pooling, Dense 512, Dropout
  - Couche finale softmax sur 120 classes

- **MobileNetV2**
  - Modèle léger pré‑entraîné
  - Adapté aux environnements **contraints (mobile / embarqué)**

- **DINOv3 ViT-B/16** (PyTorch)
  - Backbone de type Vision Transformer reconstruit
  - Classification via un MLP (Dropout + Linear)
  - Utilise des **poids pré‑entraînés** chargés depuis un `.pt`

---

## Chargement des modèles & artefacts

- **Stockage des artefacts**
  - Poids des modèles (`.h5`, `.pt`)
  - Encodeurs de labels (`label_encoder_*.pkl`)
  - Hébergés sur **HuggingFace Hub** (`LudGold/P07_POC`)

- **Fonctionnement**
  - Au lancement de l’app :
    - Téléchargement des fichiers manquants
    - Mise en cache avec `@st.cache_resource`
  - Gestion des **erreurs de téléchargement** (messages Streamlit)

- **Avantages**
  - Application **portable**
  - Séparation claire entre **code** et **modèles**

---

## Interface Streamlit

- Page unique avec 3 onglets :
  - **Exploration des données**
  - **Identification (prédiction)**
  - **Performance des modèles**

- **Barre latérale**
  - Présentation des **3 modèles** et de leur **statut de chargement**
  - Rappel du **dataset** et du **split**

- **Accessibilité**
  - Palette de couleurs compatibles **WCAG 2.1**
  - Mises en forme lisibles (titres, métriques, légendes Plotly)

---

## Onglet « Identification (prédiction) »

- Étapes pour l’utilisateur :
  1. Choisir un **modèle** (ConvNeXt, MobileNet, DINOv3).
  2. Charger une **photo** (JPG, JPEG, PNG).
  3. Cliquer sur **« Lancer l’identification »**.

- Résultats affichés :
  - **Race prédite Top‑1** (nom formaté, ex. “Golden Retriever”).
  - **Confiance** (en pourcentage) avec message :
    - Succès si \(\text{conf} \geq 50\%\)
    - Avertissement si confiance **faible**
  - **Temps d’inférence** en millisecondes.
  - **Top‑5 prédictions** sous forme de barres de progression.

---

## Comparaison des performances

- Onglet **« Performance des modèles »**
  - Tableau récapitulatif (jeu de test : ~3 087 images, 119 races) :
    - Accuracy Top‑1
    - Accuracy Top‑5
    - Nombre de paramètres (M)
    - Temps d’inférence moyen

- Résultats (exemple des ordres de grandeur) :
  - **ConvNeXt-Tiny**
    - Accuracy ≈ 88,5 %, Top‑5 ≈ 99,2 %
    - Bon compromis **précision / vitesse**
  - **MobileNetV2**
    - Accuracy ≈ 76 %, Top‑5 ≈ 96 %
    - Modèle le plus **léger** et rapide
  - **DINOv3 ViT-B/16**
    - Accuracy ≈ 88 %, Top‑5 ≈ 98,7 %
    - Modèle **plus lourd**, mais très performant

---

## Synthèse & perspectives

- **Apports du POC**
  - Démonstration de la **faisabilité** de l’identification de races de chien.
  - Interface **simple** et **pédagogique** pour les non‑experts.
  - Comparaison claire de plusieurs **architectures de Deep Learning**.

- **Perspectives**
  - Déploiement sur un **serveur interne** ou le **cloud**.
  - Amélioration de la **gestion d’images en conditions réelles** (smartphone).
  - Intégration d’une **base de données** pour historiser les prédictions.
  - Optimisation pour un usage **mobile / tablette**.

---

<!-- _class: lead -->

# Merci de votre attention

**Questions ?**  
Application : **Le Refuge — Identification de races de chiens**  
Auteur : **Ludivine Goldstein**

