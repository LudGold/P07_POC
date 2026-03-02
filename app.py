import streamlit as st
import PIL.Image as Image
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
import time

# --- CONFIGURATION & ACCESSIBILITÉ ---
st.set_page_config(
    page_title="Le Refuge - Identification Chiens", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# WCAG 1: Titres clairs et structure hiérarchique
st.title(" Assistant d'Indexation - Le Refuge")
st.markdown("### Classification de races de chiens avec Deep Learning")

# --- CHARGEMENT DES MODÈLES ET LABEL ENCODERS ---
@st.cache_resource
def load_models_and_encoders():
    """Charge les 3 modèles et leurs label encoders"""
    models = {}
    encoders = {}
    
    try:
        # ConvNeXt (TensorFlow)
        models['convnext'] = load_model('best_convnext_120races.h5')
        with open('label_encoder_convnext.pkl', 'rb') as f:
            encoders['convnext'] = pickle.load(f)
        st.sidebar.success(" ConvNeXt chargé")
    except Exception as e:
        st.sidebar.error(f" ConvNeXt: {e}")
    
    try:
        # MobileNetV2 (TensorFlow)
        models['mobilenet'] = load_model('best_mobilenet_120races.h5')
        with open('label_encoder_mobilenet.pkl', 'rb') as f:
            encoders['mobilenet'] = pickle.load(f)
        st.sidebar.success(" MobileNetV2 chargé")
    except Exception as e:
        st.sidebar.error(f" MobileNetV2: {e}")
    
    try:
        # DINOv2 (PyTorch)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Définir la classe du modèle (copie de votre code)
        class DINOv3Classifier(nn.Module):
            def __init__(self, backbone, hidden_size: int, num_classes: int):
                super().__init__()
                self.backbone = backbone
                self.head = nn.Sequential(
                    nn.Linear(hidden_size, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, num_classes)
                )
            
            def forward(self, x):
                with torch.no_grad():
                    outputs = self.backbone(x, output_hidden_states=True)
                    features = outputs.hidden_states[-1][:, 0, :]
                return self.head(features)
        
        # Charger le backbone DINOv2
        backbone = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
        
        with open('label_encoder_dinov3.pkl', 'rb') as f:
            encoders['dinov2'] = pickle.load(f)
        
        num_classes = len(encoders['dinov2'].classes_)
        model_dinov2 = DINOv3Classifier(backbone, hidden_size=768, num_classes=num_classes)
        model_dinov2.load_state_dict(torch.load('best_dinov3_120races.pt', map_location=device))
        model_dinov2.to(device)
        model_dinov2.eval()
        models['dinov2'] = model_dinov2
        
        # Processeur d'images
        processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
        norm_mean = processor.image_mean if hasattr(processor, 'image_mean') else [0.485, 0.456, 0.406]
        norm_std = processor.image_std if hasattr(processor, 'image_std') else [0.229, 0.224, 0.225]
        models['dinov2_transform'] = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
        models['dinov2_device'] = device
        
        st.sidebar.success(" DINOv2 chargé")
    except Exception as e:
        st.sidebar.error(f" DINOv2: {e}")
    
    return models, encoders

# --- FONCTIONS DE PRÉDICTION ---
def predict_convnext(image, models, encoders):
    """Prédiction avec ConvNeXt"""
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = convnext_preprocess(img_array)
    
    start = time.time()
    predictions = models['convnext'].predict(img_array, verbose=0)
    inference_time = time.time() - start
    
    top5_idx = np.argsort(predictions[0])[-5:][::-1]
    top5_probs = predictions[0][top5_idx]
    top5_labels = encoders['convnext'].inverse_transform(top5_idx)
    
    return top5_labels, top5_probs, inference_time

def predict_mobilenet(image, models, encoders):
    """Prédiction avec MobileNetV2"""
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = mobilenet_preprocess(img_array)
    
    start = time.time()
    predictions = models['mobilenet'].predict(img_array, verbose=0)
    inference_time = time.time() - start
    
    top5_idx = np.argsort(predictions[0])[-5:][::-1]
    top5_probs = predictions[0][top5_idx]
    top5_labels = encoders['mobilenet'].inverse_transform(top5_idx)
    
    return top5_labels, top5_probs, inference_time

def predict_dinov2(image, models, encoders):
    """Prédiction avec DINOv2"""
    device = models['dinov2_device']
    transform = models['dinov2_transform']
    model = models['dinov2']
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    start = time.time()
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    inference_time = time.time() - start
    
    top5_idx = np.argsort(probs)[-5:][::-1]
    top5_probs = probs[top5_idx]
    top5_labels = encoders['dinov2'].inverse_transform(top5_idx)
    
    return top5_labels, top5_probs, inference_time

# --- CHARGEMENT DES MODÈLES ---
with st.spinner(' Chargement des modèles...'):
    models, encoders = load_models_and_encoders()

# --- SIDEBAR : INFORMATIONS SUR LES MODÈLES ---
with st.sidebar:
    st.header(" Informations")
    
    st.subheader("Modèles disponibles")
    st.markdown("""
    - **ConvNeXt-Tiny** (Meta AI, 2022)
    - **MobileNetV2** (Google, 2018)  
    - **DINOv2 ViT-B/16** (Meta AI, 2025)
    """)
    
    st.subheader("Dataset")
    st.markdown("""
    - **Stanford Dogs**
    - 120 races de chiens
    - ~20 580 images
    - Split: 70% train, 15% val, 15% test
    """)

# --- SECTION 1 : EXPLORATION (EDA) ---
with st.expander(" Analyse de la base de données", expanded=False):
    st.write("Le dataset Stanford Dogs contient 20,580 images réparties sur 120 races.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total d'images", "20,580")
    with col2:
        st.metric("Nombre de races", "120")
    with col3:
        st.metric("Images par race (moy.)", "~172")
    
    st.info(" Les images ont été réparties de manière stratifiée entre les ensembles train/val/test.")

# --- SECTION 2 : MOTEUR DE PRÉDICTION ---
st.header(" Identifier un pensionnaire")

# WCAG 3: Labels de formulaire clairs
uploaded_file = st.file_uploader(
    "Choisissez une photo de chien...", 
    type=["jpg", "png", "jpeg"],
    help="Formats supportés: JPG, PNG, JPEG"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(image, caption="Image chargée", use_container_width=True)
    
    with col2:
        st.subheader("Sélectionnez le(s) modèle(s) à utiliser")
        
        use_convnext = st.checkbox("ConvNeXt-Tiny", value=True, disabled='convnext' not in models)
        use_mobilenet = st.checkbox("MobileNetV2", value=True, disabled='mobilenet' not in models)
        use_dinov2 = st.checkbox("DINOv2 ViT-B/16", value=True, disabled='dinov2' not in models)
        
        if st.button(" Analyser l'image", type="primary"):
            results = []
            
            # ConvNeXt
            if use_convnext and 'convnext' in models:
                with st.spinner('Analyse par ConvNeXt en cours...'):
                    labels, probs, inf_time = predict_convnext(image, models, encoders)
                    results.append({
                        'model': 'ConvNeXt-Tiny',
                        'label': labels[0],
                        'confidence': probs[0],
                        'top5_labels': labels,
                        'top5_probs': probs,
                        'inference_time': inf_time
                    })
            
            # MobileNetV2
            if use_mobilenet and 'mobilenet' in models:
                with st.spinner('Analyse par MobileNetV2 en cours...'):
                    labels, probs, inf_time = predict_mobilenet(image, models, encoders)
                    results.append({
                        'model': 'MobileNetV2',
                        'label': labels[0],
                        'confidence': probs[0],
                        'top5_labels': labels,
                        'top5_probs': probs,
                        'inference_time': inf_time
                    })
            
            # DINOv2
            if use_dinov2 and 'dinov2' in models:
                with st.spinner('Analyse par DINOv2 en cours...'):
                    labels, probs, inf_time = predict_dinov2(image, models, encoders)
                    results.append({
                        'model': 'DINOv2 ViT-B/16',
                        'label': labels[0],
                        'confidence': probs[0],
                        'top5_labels': labels,
                        'top5_probs': probs,
                        'inference_time': inf_time
                    })
            
            # Affichage des résultats
            if results:
                st.success(" Analyse terminée !")
                
                # Résultats principaux
                st.subheader(" Prédictions")
                cols = st.columns(len(results))
                
                for i, result in enumerate(results):
                    with cols[i]:
                        st.markdown(f"**{result['model']}**")
                        st.metric("Race prédite", result['label'].title())
                        st.metric("Confiance", f"{result['confidence']:.1%}")
                        st.caption(f"⏱️ Temps: {result['inference_time']*1000:.0f}ms")
                
                # Top-5 prédictions détaillées
                st.subheader(" Top-5 prédictions par modèle")
                
                for result in results:
                    with st.expander(f"{result['model']} - Détails"):
                        df = pd.DataFrame({
                            'Race': [label.title() for label in result['top5_labels']],
                            'Confiance': [f"{prob:.2%}" for prob in result['top5_probs']],
                            'Score': result['top5_probs']
                        })
                        
                        # Afficher le dataframe
                        st.dataframe(
                            df[['Race', 'Confiance']], 
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Barre de progression pour chaque prédiction
                        for idx, row in df.iterrows():
                            st.progress(float(row['Score']), text=f"{row['Race']}: {row['Confiance']}")
                
                # Comparaison des modèles
                if len(results) > 1:
                    st.subheader(" Comparaison des modèles")
                    
                    comparison_df = pd.DataFrame({
                        'Modèle': [r['model'] for r in results],
                        'Race prédite': [r['label'].title() for r in results],
                        'Confiance': [f"{r['confidence']:.2%}" for r in results],
                        'Temps (ms)': [f"{r['inference_time']*1000:.0f}" for r in results]
                    })
                    
                    st.dataframe(comparison_df, hide_index=True, use_container_width=True)
                    
                    # Consensus
                    predicted_labels = [r['label'] for r in results]
                    if len(set(predicted_labels)) == 1:
                        st.success(f" **Consensus**: Tous les modèles s'accordent sur **{predicted_labels[0].title()}**")
                    else:
                        st.warning(f" **Divergence**: Les modèles ont prédit des races différentes")

# --- SECTION 3 : COMPARAISON DES MODÈLES (PERFORMANCES GLOBALES) ---
st.header(" Performance des Modèles")

# Données de performance (à remplacer par vos vrais résultats)
# Vous devrez les extraire de vos fichiers convnext_results.txt, dinov3_results.txt, etc.
performance_data = {
    'Modèle': ['ConvNeXt-Tiny', 'MobileNetV2', 'DINOv2 ViT-B/16'],
    'Test Accuracy': [0.89, 0.76, 0.88],  
    'Test Top-5 Accuracy': [0.99, 0.96, 0.98],  
    'Paramètres (M)': [28.6, 3.5, 86.0],  
    'Temps d\'inférence (ms)': [45, 25, 120]  # À remplacer - pas calculé comme metrique 
}

performance_df = pd.DataFrame(performance_data)

col1, col2 = st.columns(2)

with col1:
    st.subheader(" Précision sur Test Set")
    st.dataframe(
        performance_df[['Modèle', 'Test Accuracy', 'Test Top-5 Accuracy']], 
        hide_index=True,
        use_container_width=True
    )

with col2:
    st.subheader(" Efficacité")
    st.dataframe(
        performance_df[['Modèle', 'Paramètres (M)', 'Temps d\'inférence (ms)']], 
        hide_index=True,
        use_container_width=True
    )

# Interprétation
st.info("""
 **Interprétation**:
- **DINOv2**: Meilleure précision mais plus lent et lourd
- **MobileNetV2**: Le plus rapide et léger, bon pour mobile
- **ConvNeXt**: Bon compromis précision/vitesse
""")

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🐾 <strong>Le Refuge</strong> - Assistant d'identification de races de chiens</p>
    <p style='font-size: 0.9em'>Outil conçu pour l'accessibilité : navigation au clavier supportée (WCAG 2.1)</p>
</div>
""", unsafe_allow_html=True)
