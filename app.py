import streamlit as st
from PIL import Image, ImageFilter, ImageOps, ImageFile, UnidentifiedImageError
import numpy as np
import pandas as pd
import pickle
import time
import os
from glob import glob
import plotly.express as px
import plotly.graph_objects as go

import tensorflow as tf
from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from types import SimpleNamespace
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Le Refuge - Identification Chiens", layout="wide")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# =====================================================================
# ARCHITECTURE DINOv3 (PyTorch)
# Le backbone est reconstruit ici car le modèle pré-entraîné original
# est privé sur HuggingFace. Les poids sont chargés depuis le .pt.
# =====================================================================

class _LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.lambda1


class _Attention(nn.Module):
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
        reshape = lambda t: t.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k, v = reshape(self.q_proj(x)), reshape(self.k_proj(x)), reshape(self.v_proj(x))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        out = attn.softmax(dim=-1) @ v
        return self.o_proj(out.transpose(1, 2).reshape(B, N, C))


class _MLP(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.up_proj = nn.Linear(dim, mlp_dim)
        self.down_proj = nn.Linear(mlp_dim, dim)

    def forward(self, x):
        return self.down_proj(F.gelu(self.up_proj(x)))


class _Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = _Attention(dim, num_heads)
        self.layer_scale1 = _LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, mlp_dim)
        self.layer_scale2 = _LayerScale(dim)

    def forward(self, x):
        x = x + self.layer_scale1(self.attention(self.norm1(x)))
        x = x + self.layer_scale2(self.mlp(self.norm2(x)))
        return x


class _Embeddings(nn.Module):
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
        self.embeddings = _Embeddings(dim, patch_size, num_register_tokens)
        self.layer = nn.ModuleList([_Block(dim, num_heads, mlp_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, pixel_values, **kwargs):
        x = self.embeddings(pixel_values)
        for blk in self.layer:
            x = blk(x)
        x = self.norm(x)
        return SimpleNamespace(pooler_output=x[:, 0])


class DINOv3Classifier(nn.Module):
    def __init__(self, backbone, hidden_size, num_classes):
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
        features = self.backbone(pixel_values=pixel_values).pooler_output
        return self.classifier(features)


# =====================================================================
# TELECHARGEMENT & CHARGEMENT DES MODELES
# =====================================================================

HF_REPO_ID = "LudGold/P07_POC"
IMAGES_LOCAL = os.path.isdir("Images")

MODEL_FILES = [
    "best_convnext_120races.h5",
    "best_mobilenet_120races.h5",
    "best_dinov3_120races.pt",
    "label_encoder_convnext.pkl",
    "label_encoder_mobilenet.pkl",
    "label_encoder_dinov3.pkl",
]


@st.cache_resource
def download_artifacts():
    """Télécharge les modèles depuis HuggingFace s'ils n'existent pas localement."""
    for fname in MODEL_FILES:
        if not os.path.exists(fname):
            try:
                path = hf_hub_download(repo_id=HF_REPO_ID, filename=fname)
                os.symlink(path, fname)
            except Exception as e:
                st.warning(f"Impossible de télécharger {fname} : {e}")


def _load_encoder(name):
    with open(f'label_encoder_{name}.pkl', 'rb') as f:
        return pickle.load(f)


@st.cache_resource
def load_all_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, encoders = {}, {}

    # ConvNeXt
    try:
        encoders['convnext'] = _load_encoder('convnext')
        n_classes = len(encoders['convnext'].classes_)
        base = ConvNeXtTiny(weights=None, include_top=False, input_shape=(224, 224, 3))
        x = GlobalAveragePooling2D(name='avg_pool')(base.output)
        x = Dropout(0.5, name='dropout')(x)
        x = Dense(512, activation='relu', name='fc1')(x)
        x = Dropout(0.3, name='dropout2')(x)
        out = Dense(n_classes, activation='softmax', name='predictions')(x)
        model = Model(inputs=base.input, outputs=out)
        model.load_weights('best_convnext_120races.h5')
        models['convnext'] = model
    except Exception as e:
        st.error(f"Erreur ConvNeXt : {e}")

    # MobileNetV2
    try:
        models['mobilenet'] = tf.keras.models.load_model('best_mobilenet_120races.h5')
        encoders['mobilenet'] = _load_encoder('mobilenet')
    except Exception as e:
        st.error(f"Erreur MobileNet : {e}")

    # DINOv3 — architecture déduite automatiquement du fichier .pt
    try:
        encoders['dinov3'] = _load_encoder('dinov3')
        sd = torch.load('best_dinov3_120races.pt', map_location=device)

        dim = sd['backbone.norm.weight'].shape[0]
        num_reg = sd['backbone.embeddings.register_tokens'].shape[1]
        patch_sz = sd['backbone.embeddings.patch_embeddings.weight'].shape[2]
        mlp_dim = sd['backbone.layer.0.mlp.up_proj.weight'].shape[0]
        n_layers = max(int(k.split('.')[2]) for k in sd if k.startswith('backbone.layer.')) + 1

        backbone = DINOv3Backbone(
            dim=dim, num_layers=n_layers, num_heads=dim // 64,
            mlp_dim=mlp_dim, patch_size=patch_sz, num_register_tokens=num_reg,
        )
        model = DINOv3Classifier(backbone, dim, len(encoders['dinov3'].classes_))
        model.load_state_dict(sd)
        model.to(device).eval()
        models['dinov3'] = model
    except Exception as e:
        st.error(f"Erreur DINOv3 : {e}")

    return {'models': models, 'encoders': encoders, 'device': device}


# =====================================================================
# FONCTIONS UTILITAIRES
# =====================================================================
# pour accessibilité
WCAG_COLORS = ["#0056B3", "#D4380D", "#1A7F37", "#6F42C1", "#B8600A"]

MODEL_DISPLAY = {
    'convnext': 'ConvNeXt-Tiny',
    'mobilenet': 'MobileNetV2',
    'dinov3': 'DINOv3 ViT-B/16',
}


@st.cache_data
def load_eda_data():
    """Charge les métadonnées EDA (local ou CSV)."""
    if IMAGES_LOCAL:
        records = []
        for folder in sorted(os.listdir("Images")):
            full = os.path.join("Images", folder)
            if not os.path.isdir(full):
                continue
            breed = folder.split('-', 1)[-1].replace('_', ' ')
            for p in glob(os.path.join(full, "*.jpg")):
                try:
                    with Image.open(p) as im:
                        w, h = im.size
                except (UnidentifiedImageError, OSError):
                    continue
                records.append({
                    "breed": breed, "folder": folder,
                    "filename": os.path.basename(p), "width": w, "height": h,
                })
        return pd.DataFrame(records)

    if os.path.isfile("eda_metadata.csv"):
        return pd.read_csv("eda_metadata.csv")

    return pd.DataFrame()


def download_breed_images(folder, filenames, max_imgs=3):
    """Télecharge quelques images d'une race depuis HuggingFace."""
    images = []
    for fname in filenames[:max_imgs]:
        try:
            path = hf_hub_download(repo_id=HF_REPO_ID, filename=f"Images_sample/{folder}/{fname}")
            with Image.open(path) as im:
                images.append(im.convert('RGB'))
        except Exception:
            pass
    return images


def get_breed_images(breed_subset):
    """Renvoie jusqu'à 5 images PIL pour une race donnée (local ou cloud)."""
    if IMAGES_LOCAL:
        paths = [os.path.join("Images", row['folder'], row['filename'])
                 for _, row in breed_subset.head(3).iterrows()]
        out = []
        for p in paths:
            try:
                with Image.open(p) as im:
                    out.append(im.convert('RGB'))
            except (UnidentifiedImageError, OSError):
                continue
        return out
    folder = breed_subset['folder'].iloc[0]
    fnames = breed_subset['filename'].tolist()
    return download_breed_images(folder, fnames, max_imgs=3)


def show_transforms(img):
    """Affiche une image et ses 3 transformations côte à côte."""
    img_224 = img.resize((224, 224))
    cols = st.columns(4)
    with cols[0]:
        st.image(img_224, caption="Originale (224x224)", use_container_width=True)
    with cols[1]:
        st.image(ImageOps.equalize(img_224), caption="Egalisation histogramme", use_container_width=True)
    with cols[2]:
        st.image(img_224.filter(ImageFilter.GaussianBlur(3)), caption="Flou gaussien", use_container_width=True)
    with cols[3]:
        st.image(img_224.filter(ImageFilter.FIND_EDGES), caption="Détection de contours", use_container_width=True)


def predict_top5(image, model_name, assets):
    """Lance la prediction et renvoie (labels, probas, temps) du top 5."""
    encoder = assets['encoders'][model_name]

    if model_name in ('convnext', 'mobilenet'):
        img_array = np.expand_dims(np.array(image.resize((224, 224))).astype('float32'), 0)
        preprocess = convnext_preprocess if model_name == 'convnext' else mobilenet_preprocess

        t0 = time.time()
        # Evite model.predict() (qui passe par la data-adapter interne) et appelle directement le modèle
        # en mode inférence.
        x = preprocess(img_array).astype('float32', copy=False)
        pred = assets['models'][model_name](x, training=False)
        preds = pred.numpy()[0]
        dt = time.time() - t0
    else:
        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tensor = tfm(image).unsqueeze(0).to(assets['device'])

        t0 = time.time()
        with torch.no_grad():
            preds = F.softmax(assets['models']['dinov3'](tensor), dim=1)[0].cpu().numpy()
        dt = time.time() - t0

    top5 = np.argsort(preds)[-5:][::-1]
    return encoder.classes_[top5], preds[top5], dt


# =====================================================================
# INTERFACE
# =====================================================================

st.title("Le Refuge — Identification de Races de Chiens")
st.caption("Classification par Deep Learning sur le dataset Stanford Dogs")

with st.spinner("Chargement des modèles..."):
    download_artifacts()
    assets = load_all_models()

available_models = list(assets['models'].keys())

# --- Sidebar ---
with st.sidebar:
    st.markdown(
        "**Le Refuge** utilise 3 modèles de Deep Learning "
        "pour identifier la race d'un chien à partir d'une photo."
    )
    st.subheader("Modèles")
    for key, label in MODEL_DISPLAY.items():
        status = "Charge" if key in assets['models'] else "Indisponible"
        st.markdown(f"- {label} — *{status}*")

    st.subheader("Dataset")
    st.markdown(
        "- **Stanford Dogs**\n"
        "- 120 races · ~20 580 images\n"
        "- Split : 70% train / 15% val / 15% test"
    )

# --- Onglets ---
tab_eda, tab_predict, tab_perf = st.tabs([
    "Exploration des données",
    "Identification (prédiction)",
    "Performance des modèles",
])

# ====================== ONGLET 1 : EDA ======================
with tab_eda:
    st.header("Analyse exploratoire du dataset Stanford Dogs")
    st.markdown(
        "Source : [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) "
        "— 20 580 images, 120 races."
    )
    eda_df = load_eda_data()

    if eda_df.empty:
        st.warning("Aucune donnée disponible (ni dossier Images/ ni eda_metadata.csv).")
    else:
        breed_counts = eda_df['breed'].value_counts().sort_values(ascending=False)

        c1, c2, c3 = st.columns(3)
        c1.metric("Images totales", f"{len(eda_df):,}")
        c2.metric("Races", f"{breed_counts.shape[0]}")
        c3.metric("Moyenne / race", f"{len(eda_df) // breed_counts.shape[0]}")

        st.markdown("---")

        # Graphique 1 : distribution
        st.subheader("Distribution du nombre d'images par race")
        fig1 = px.bar(
            x=breed_counts.index, y=breed_counts.values,
            labels={"x": "Race", "y": "Nombre d'images"},
            color_discrete_sequence=[WCAG_COLORS[0]],
        )
        fig1.update_layout(xaxis_tickangle=-45, height=500, margin=dict(b=160))
        fig1.add_hline(
            y=breed_counts.mean(), line_dash="dash", line_color=WCAG_COLORS[1],
            annotation_text=f"Moyenne : {breed_counts.mean():.0f}",
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Graphique 2 : dimensions
        st.subheader("Dimensions des images")
        sample = eda_df.sample(min(500, len(eda_df)), random_state=42)
        fig2 = go.Figure()
        fig2.add_trace(go.Box(y=sample['width'], name="Largeur (px)", marker_color=WCAG_COLORS[0]))
        fig2.add_trace(go.Box(y=sample['height'], name="Hauteur (px)", marker_color=WCAG_COLORS[2]))
        fig2.update_layout(yaxis_title="Pixels", height=400)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")

        # Exemples d'images
        st.subheader("Exemples d'images par race")
        selected_breed = st.selectbox(
            "Choisir une race", sorted(eda_df['breed'].unique()),
            help="Sélectionnez une race pour voir des images du dataset.",
        )
        breed_subset = eda_df[eda_df['breed'] == selected_breed]
        st.caption(f"{len(breed_subset)} images pour cette race.")

        with st.spinner("Chargement des images..."):
            breed_imgs = get_breed_images(breed_subset)

        if breed_imgs:
            cols = st.columns(len(breed_imgs))
            for i, img in enumerate(breed_imgs):
                with cols[i]:
                    st.image(img, caption=f"#{i+1}", use_container_width=True)

            st.markdown("---")
            st.subheader("Transformations appliquées")
            st.markdown("Pré-traitements possibles : égalisation, flou gaussien, détection de contours.")
            show_transforms(breed_imgs[0])

# ====================== ONGLET 2 : PREDICTION ======================
with tab_predict:
    st.header("Identifier la race d'un chien")
    st.markdown("Soumettez une photo et sélectionnez un modèle.")

    col_upload, col_config = st.columns([2, 1])

    with col_config:
        st.subheader("Configuration")
        if not available_models:
            st.error("Aucun modèle n'a pu être chargé.")
        else:
            selected_model = st.selectbox(
                "Modèle", available_models,
                format_func=lambda k: MODEL_DISPLAY.get(k, k),
                help="Choisissez le modèle à utiliser.",
            )

    with col_upload:
        uploaded_file = st.file_uploader(
            "Photo de chien", type=["jpg", "jpeg", "png"],
            help="Formats : JPG, JPEG, PNG. Taille max : 5 Mo.",
        )

    if uploaded_file and available_models:
        try:
            img = Image.open(uploaded_file).convert('RGB')
        except (UnidentifiedImageError, OSError) as e:
            st.error(f"Impossible de lire ce fichier comme une image: {e}")
            img = None
            uploaded_file = None
        col_img, col_res = st.columns([1, 2])

        with col_img:
            st.image(img, caption="Image soumise", use_container_width=True)

        with col_res:
            if st.button("Lancer l'identification", type="primary"):
                with st.spinner(f"Analyse avec {MODEL_DISPLAY[selected_model]}..."):
                    labels, probs, dt = predict_top5(img, selected_model, assets)

                name = labels[0].replace('_', ' ').title()
                conf = probs[0]

                if conf >= 0.5:
                    st.success(f"**{name}** — confiance {conf:.1%}")
                else:
                    st.warning(f"**{name}** — confiance faible ({conf:.1%})")

                st.caption(f"Temps d'inférence : {dt*1000:.0f} ms")

                st.markdown("#### Top-5 prédictions")
                for i in range(5):
                    st.progress(float(probs[i]),
                                text=f"{labels[i].replace('_', ' ').title()} — {probs[i]:.1%}")

# ====================== ONGLET 3 : PERFORMANCE ======================
with tab_perf:
    st.header("Comparaison des performances")
    st.markdown("Résultats sur le jeu de test (3 087 images, 119 races).")

    perf = pd.DataFrame([
        {"Modèle": "ConvNeXt-Tiny", "Accuracy": 0.8853, "Top-5": 0.9922,
         "Params (M)": 28.3, "Inférence": "~45 ms"},
        {"Modèle": "MobileNetV2", "Accuracy": 0.76, "Top-5": 0.96,
         "Params (M)": 3.5, "Inférence": "~25 ms"},
        {"Modèle": "DINOv3 ViT-B/16", "Accuracy": 0.8801, "Top-5": 0.9874,
         "Params (M)": 86.1, "Inférence": "~120 ms"},
    ])
    st.dataframe(perf, hide_index=True, use_container_width=True)

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=perf['Modèle'], y=perf['Accuracy'], name='Top-1',
        marker_color=WCAG_COLORS[0],
        text=[f"{v:.1%}" for v in perf['Accuracy']], textposition='outside',
    ))
    fig3.add_trace(go.Bar(
        x=perf['Modèle'], y=perf['Top-5'], name='Top-5',
        marker_color=WCAG_COLORS[2],
        text=[f"{v:.1%}" for v in perf['Top-5']], textposition='outside',
    ))
    fig3.update_layout(barmode='group', yaxis_range=[0, 1.12], height=450)
    st.plotly_chart(fig3, use_container_width=True)

    st.info(
        "**Synthese** : ConvNeXt offre le meilleur compromis précision/vitesse. "
        "DINOv3 est proche en précision mais plus lourd. "
        "MobileNetV2 est le plus léger, idéal pour le mobile."
    )

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#555; font-size:0.9em'>"
    "Le Refuge — Projet P07 · LudGold — Accessibilité WCAG 2.1"
    "</div>",
    unsafe_allow_html=True,
)
