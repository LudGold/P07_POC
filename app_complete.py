import streamlit as st
import PIL.Image as Image
from PIL import ImageFilter, ImageOps
import numpy as np
import pandas as pd
import pickle
import time
import os
from glob import glob
import plotly.express as px
import plotly.graph_objects as go

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.applications import ConvNeXtTiny, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# PyTorch (DINOv3)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from types import SimpleNamespace

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Le Refuge - Identification Chiens", layout="wide")

# --- ARCHITECTURE PYTORCH (DINOv3 - reconstruction locale, sans HuggingFace) ---

class _DINOv3LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.lambda1


class _DINOv3Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return self.o_proj((attn @ v).transpose(1, 2).reshape(B, N, C))


class _DINOv3MLP(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.up_proj = nn.Linear(dim, mlp_dim)
        self.down_proj = nn.Linear(mlp_dim, dim)

    def forward(self, x):
        return self.down_proj(F.gelu(self.up_proj(x)))


class _DINOv3Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = _DINOv3Attention(dim, num_heads)
        self.layer_scale1 = _DINOv3LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _DINOv3MLP(dim, mlp_dim)
        self.layer_scale2 = _DINOv3LayerScale(dim)

    def forward(self, x):
        x = x + self.layer_scale1(self.attention(self.norm1(x)))
        x = x + self.layer_scale2(self.mlp(self.norm2(x)))
        return x


class _DINOv3Embeddings(nn.Module):
    def __init__(self, dim, patch_size, num_register_tokens):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, dim))

    def forward(self, pixel_values):
        B = pixel_values.shape[0]
        patches = self.patch_embeddings(pixel_values).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        reg = self.register_tokens.expand(B, -1, -1)
        return torch.cat([cls, reg, patches], dim=1)


class DINOv3Backbone(nn.Module):
    def __init__(self, dim=768, num_layers=12, num_heads=12, mlp_dim=3072,
                 patch_size=16, num_register_tokens=4):
        super().__init__()
        self.embeddings = _DINOv3Embeddings(dim, patch_size, num_register_tokens)
        self.layer = nn.ModuleList([
            _DINOv3Block(dim, num_heads, mlp_dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, pixel_values, **kwargs):
        x = self.embeddings(pixel_values)
        for blk in self.layer:
            x = blk(x)
        x = self.norm(x)
        return SimpleNamespace(pooler_output=x[:, 0])


class DINOv3Classifier(nn.Module):
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
        return self.classifier(outputs.pooler_output)

# --- CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def load_all_models():
    assets = {'models': {}, 'encoders': {}, 'device': torch.device("cpu")}
    if torch.cuda.is_available():
        assets['device'] = torch.device("cuda")

    # 1. CONVNEXT (Reconstruction identique au script d'entraînement)
    try:
        with open('label_encoder_convnext.pkl', 'rb') as f:
            assets['encoders']['convnext'] = pickle.load(f)
        num_classes_conv = len(assets['encoders']['convnext'].classes_)

        base_conv = ConvNeXtTiny(weights=None, include_top=False, input_shape=(224, 224, 3))
        x = GlobalAveragePooling2D(name='avg_pool')(base_conv.output)
        x = Dropout(0.5, name='dropout')(x)
        x = Dense(512, activation='relu', name='fc1')(x)
        x = Dropout(0.3, name='dropout2')(x)
        out = Dense(num_classes_conv, activation='softmax', name='predictions')(x)
        m_conv = Model(inputs=base_conv.input, outputs=out)
        m_conv.load_weights('best_convnext_120races.h5')
        assets['models']['convnext'] = m_conv
    except Exception as e:
        st.error(f"Erreur ConvNeXt : {e}")

    # 2. MOBILENETV2
    try:
        # MobileNet est standard, load_model devrait fonctionner
        assets['models']['mobilenet'] = tf.keras.models.load_model('best_mobilenet_120races.h5')
        with open('label_encoder_mobilenet.pkl', 'rb') as f:
            assets['encoders']['mobilenet'] = pickle.load(f)
    except Exception as e:
        st.error(f"Erreur MobileNet : {e}")

    # 3. DINOv3 (PyTorch) — backbone reconstruit localement
    try:
        with open('label_encoder_dinov3.pkl', 'rb') as f:
            le_dino = pickle.load(f)
            assets['encoders']['dinov3'] = le_dino

        state_dict = torch.load('best_dinov3_120races.pt', map_location=assets['device'])
        dim = int(state_dict['backbone.norm.weight'].shape[0])
        num_reg = int(state_dict['backbone.embeddings.register_tokens'].shape[1])
        patch_sz = int(state_dict['backbone.embeddings.patch_embeddings.weight'].shape[2])
        mlp_dim = int(state_dict['backbone.layer.0.mlp.up_proj.weight'].shape[0])
        num_layers = max(int(k.split('.')[2]) for k in state_dict if k.startswith('backbone.layer.')) + 1
        num_heads = dim // 64

        backbone = DINOv3Backbone(
            dim=dim, num_layers=num_layers, num_heads=num_heads,
            mlp_dim=mlp_dim, patch_size=patch_sz, num_register_tokens=num_reg
        )
        m_dino = DINOv3Classifier(backbone, dim, len(le_dino.classes_))
        m_dino.load_state_dict(state_dict)
        m_dino.to(assets['device']).eval()
        assets['models']['dinov3'] = m_dino
    except Exception as e:
        st.error(f"Erreur DINOv3 : {e}")

    return assets

# --- PALETTE WCAG (contraste >= 4.5:1 sur fond blanc) ---
WCAG_COLORS = ["#0056B3", "#D4380D", "#1A7F37", "#6F42C1", "#B8600A",
               "#0E606B", "#A6266E", "#4A6FA5", "#8B6C42", "#2E5339"]

MODEL_DISPLAY = {
    'convnext': 'ConvNeXt-Tiny (Meta AI, 2022)',
    'mobilenet': 'MobileNetV2 (Google, 2018)',
    'dinov3': 'DINOv3 ViT-B/16 (Meta AI, 2025)',
}

# --- FONCTIONS EDA ---
@st.cache_data
def load_eda_data():
    """Scanne le dossier Images/ et construit un DataFrame descriptif."""
    path_base = "Images"
    records = []
    for folder in sorted(os.listdir(path_base)):
        full = os.path.join(path_base, folder)
        if not os.path.isdir(full):
            continue
        breed = folder.split('-', 1)[-1].replace('_', ' ')
        imgs = glob(os.path.join(full, "*.jpg"))
        for p in imgs:
            records.append({"breed": breed, "path": p, "folder": folder})
    return pd.DataFrame(records)


def show_image_transforms(img):
    """Affiche l'image originale et 3 transformations."""
    img_224 = img.resize((224, 224))
    cols = st.columns(4)
    with cols[0]:
        st.image(img_224, caption="Originale (224x224)", use_container_width=True)
    with cols[1]:
        eq = ImageOps.equalize(img_224)
        st.image(eq, caption="Egalisation histogramme", use_container_width=True)
    with cols[2]:
        blurred = img_224.filter(ImageFilter.GaussianBlur(radius=3))
        st.image(blurred, caption="Flou gaussien (r=3)", use_container_width=True)
    with cols[3]:
        edges = img_224.filter(ImageFilter.FIND_EDGES)
        st.image(edges, caption="Detection de contours", use_container_width=True)


# --- FONCTIONS DE PRÉDICTION ---
def predict_top5(image, model_name, assets):
    """Renvoie (labels_top5, probs_top5, temps_inference)."""
    img_224 = image.resize((224, 224))
    encoder = assets['encoders'][model_name]

    if model_name in ['convnext', 'mobilenet']:
        img_array = np.expand_dims(np.array(img_224).astype('float32'), 0)
        preprocess = convnext_preprocess if model_name == 'convnext' else mobilenet_preprocess
        proc_img = preprocess(img_array)

        t0 = time.time()
        preds = assets['models'][model_name].predict(proc_img, verbose=0)[0]
        dt = time.time() - t0

        top5_idx = np.argsort(preds)[-5:][::-1]
        return encoder.classes_[top5_idx], preds[top5_idx], dt

    else:  # dinov3
        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tensor = tfm(image).unsqueeze(0).to(assets['device'])

        t0 = time.time()
        with torch.no_grad():
            logits = assets['models']['dinov3'](tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
        dt = time.time() - t0

        top5_idx = np.argsort(probs)[-5:][::-1]
        return encoder.classes_[top5_idx], probs[top5_idx], dt


# ==========================================================================
# INTERFACE UTILISATEUR
# ==========================================================================

st.title("Le Refuge - Assistant d'Identification de Races de Chiens")
st.caption("Classification par Deep Learning sur le dataset Stanford Dogs")

with st.spinner("Chargement des modèles..."):
    assets = load_all_models()

available_models = list(assets['models'].keys())

# --- SIDEBAR ---
with st.sidebar:
   
    st.markdown(
        "**Le Refuge** utilise 3 modèles de Deep Learning "
        "pour identifier la race d'un chien à partir d'une photo."
    )
    st.subheader("Modèles disponibles")
    for key, label in MODEL_DISPLAY.items():
        status = "Charge" if key in assets['models'] else "Indisponible"
        icon = "+" if key in assets['models'] else "-"
        st.markdown(f"{icon} {label} — *{status}*")

    st.subheader("Dataset")
    st.markdown(
        "- **Stanford Dogs**\n"
        "- 120 races de chiens\n"
        "- ~20 580 images\n"
        "- Split : 70 % train / 15 % val / 15 % test"
    )

# --- ONGLETS PRINCIPAUX ---
tab_eda, tab_predict, tab_perf = st.tabs([
    "Exploration des données",
    "Identification (prédiction)",
    "Performance des modèles",
])

# ======================================================================
# ONGLET 1 : EXPLORATION (EDA)
# ======================================================================
with tab_eda:
    st.header("Analyse exploratoire du dataset Stanford Dogs")
    eda_df = load_eda_data()
    breed_counts = eda_df['breed'].value_counts().sort_values(ascending=False)

    # --- Metriques globales ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Nombre total d'images", f"{len(eda_df):,}")
    c2.metric("Nombre de races", f"{breed_counts.shape[0]}")
    c3.metric("Images par race (moyenne)", f"{len(eda_df) // breed_counts.shape[0]}")

    st.markdown("---")

    # --- Graphique interactif 1 : distribution des images par race ---
    st.subheader("Distribution du nombre d'images par race")
    fig_dist = px.bar(
        x=breed_counts.index,
        y=breed_counts.values,
        labels={"x": "Race", "y": "Nombre d'images"},
        color_discrete_sequence=[WCAG_COLORS[0]],
    )
    fig_dist.update_layout(
        xaxis_tickangle=-45,
        xaxis_title="Race",
        yaxis_title="Nombre d'images",
        height=500,
        margin=dict(b=160),
        font=dict(size=13),
    )
    fig_dist.add_hline(
        y=breed_counts.mean(), line_dash="dash", line_color=WCAG_COLORS[1],
        annotation_text=f"Moyenne : {breed_counts.mean():.0f}",
        annotation_font_color=WCAG_COLORS[1],
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # --- Graphique interactif 2 : boite a moustaches des dimensions ---
    st.subheader("Dimensions des images (échantillon)")
    sample_paths = eda_df.sample(min(500, len(eda_df)), random_state=42)['path']
    dims = []
    for p in sample_paths:
        try:
            w, h = Image.open(p).size
            dims.append({"Largeur": w, "Hauteur": h, "Ratio": round(w / h, 2)})
        except Exception:
            pass
    dims_df = pd.DataFrame(dims)

    fig_dims = go.Figure()
    fig_dims.add_trace(go.Box(y=dims_df['Largeur'], name="Largeur (px)",
                              marker_color=WCAG_COLORS[0]))
    fig_dims.add_trace(go.Box(y=dims_df['Hauteur'], name="Hauteur (px)",
                              marker_color=WCAG_COLORS[2]))
    fig_dims.update_layout(
        yaxis_title="Pixels",
        height=400,
        font=dict(size=13),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_dims, use_container_width=True)

    st.markdown("---")

    # --- Exemples d'images par race ---
    st.subheader("Exemples d'images par race")
    breeds_sorted = sorted(eda_df['breed'].unique())
    selected_breed = st.selectbox(
        "Choisir une race pour afficher des exemples",
        breeds_sorted,
        help="Sélectionnez une race dans la liste pour voir des images du dataset.",
    )
    breed_imgs = eda_df[eda_df['breed'] == selected_breed]['path'].tolist()
    st.caption(f"{len(breed_imgs)} images disponibles pour cette race.")
    cols = st.columns(5)
    for i, p in enumerate(breed_imgs[:5]):
        with cols[i]:
            st.image(Image.open(p), caption=f"#{i+1}", use_container_width=True)

    st.markdown("---")

    # --- Transformations d'images ---
    st.subheader("Exemples de transformations appliquées")
    st.markdown(
        "Aperçu des pré-traitements possibles sur une image du dataset "
        "(égalisation d'histogramme, flou gaussien, détection de contours)."
    )
    sample_img = Image.open(breed_imgs[0]).convert('RGB')
    show_image_transforms(sample_img)


# ======================================================================
# ONGLET 2 : PREDICTION
# ======================================================================
with tab_predict:
    st.header("Identifier un pensionnaire canin")
    st.markdown("Soumettez la photo d'un chien et sélectionnez un modèle pour identifier sa race.")

    col_upload, col_config = st.columns([2, 1])

    with col_config:
        st.subheader("Configuration")
        if not available_models:
            st.error("Aucun modèle n'a pu être chargé.")
        else:
            selected_model = st.selectbox(
                "Modèle de classification",
                available_models,
                format_func=lambda k: MODEL_DISPLAY.get(k, k),
                help="Choisissez l'un des 3 modèles à utiliser pour la prédiction.",
            )

    with col_upload:
        uploaded_file = st.file_uploader(
            "Choisissez une photo de chien",
            type=["jpg", "jpeg", "png"],
            help="Formats supportés : JPG, JPEG, PNG. Taille max recommandée : 5 Mo.",
        )

    if uploaded_file and available_models:
        img = Image.open(uploaded_file).convert('RGB')
        col_img, col_res = st.columns([1, 2])

        with col_img:
            st.image(img, caption="Image soumise", use_container_width=True)

        with col_res:
            if st.button("Lancer l'identification", type="primary"):
                with st.spinner(f"Analyse en cours avec {MODEL_DISPLAY[selected_model]}..."):
                    labels, probs, dt = predict_top5(img, selected_model, assets)

                breed_name = labels[0].replace('_', ' ').title()
                confidence = probs[0]

                if confidence >= 0.5:
                    st.success(f"**Race prédite : {breed_name}** — confiance {confidence:.1%}")
                else:
                    st.warning(f"**Race prédite : {breed_name}** — confiance faible ({confidence:.1%})")

                st.caption(f"Temps d'inférence : {dt*1000:.0f} ms")

                st.markdown("#### Top-5 prédictions")
                for i in range(5):
                    label = labels[i].replace('_', ' ').title()
                    prob = float(probs[i])
                    st.progress(prob, text=f"{label} — {prob:.1%}")


# ======================================================================
# ONGLET 3 : PERFORMANCE DES MODELES
# ======================================================================
with tab_perf:
    st.header("Comparaison des performances")
    st.markdown("Resultats évalués sur le jeu de test (3 087 images, 119 races).")

    perf = pd.DataFrame([
        {"Modele": "ConvNeXt-Tiny", "Test Accuracy": 0.8853, "Test Top-5": 0.9922,
         "Parametres (M)": 28.3, "Inference (ms)": "~45"},
        {"Modele": "MobileNetV2", "Test Accuracy": 0.76, "Test Top-5": 0.96,
         "Parametres (M)": 3.5, "Inference (ms)": "~25"},
        {"Modele": "DINOv3 ViT-B/16", "Test Accuracy": 0.8801, "Test Top-5": 0.9874,
         "Parametres (M)": 86.1, "Inference (ms)": "~120"},
    ])

    st.dataframe(perf, hide_index=True, use_container_width=True)

    # Graphique comparatif
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Bar(
        x=perf['Modele'], y=perf['Test Accuracy'], name='Top-1 Accuracy',
        marker_color=WCAG_COLORS[0],
        text=[f"{v:.1%}" for v in perf['Test Accuracy']], textposition='outside',
    ))
    fig_perf.add_trace(go.Bar(
        x=perf['Modele'], y=perf['Test Top-5'], name='Top-5 Accuracy',
        marker_color=WCAG_COLORS[2],
        text=[f"{v:.1%}" for v in perf['Test Top-5']], textposition='outside',
    ))
    fig_perf.update_layout(
        barmode='group',
        yaxis_title="Score",
        yaxis_range=[0, 1.12],
        height=450,
        font=dict(size=13),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_perf, use_container_width=True)

    st.info(
        "**Synthese** : ConvNeXt offre le meilleur compromis précision / vitesse. "
        "DINOv3 est tres proche en precision mais beaucoup plus lourd. "
        "MobileNetV2 est le plus leger et rapide, ideal pour un deploiement mobile."
    )

# --- FOOTER ---
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#555; font-size:0.9em'>"
    "Le Refuge — Assistant d'identification de races de chiens<br>"
    "Projet P07 — Accessibilité WCAG 2.1"
    "</div>",
    unsafe_allow_html=True,
)