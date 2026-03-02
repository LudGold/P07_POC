# Configuration du Cache Local pour Swin Transformer

## Problème résolu
Le modèle Swin Transformer essayait d'accéder au cache système `/home/ze-hive/.cache/huggingface/` qui a des problèmes de permissions (fichiers de verrous bloqués).

## Solution : Cache Local

### Étape 1 : Télécharger le modèle dans le cache local

```bash
cd /home/ludivine/P07_POC_LG
python download_swin_model.py
```

**Ce script va :**
- Créer le dossier `./swin_model_cache`
- Télécharger le processeur d'images (~quelques Ko)
- Télécharger le modèle complet (~100 Mo)
- Tout stocker dans votre cache local

⏱️ **Durée : 2-5 minutes** (selon votre connexion internet)

### Étape 2 : Lancer votre script principal

Une fois le téléchargement terminé :

```bash
python Swin_Transformer
```

**Le script principal va maintenant :**
- Utiliser UNIQUEMENT le cache local (`./swin_model_cache`)
- Ne JAMAIS essayer d'accéder au cache système
- Démarrer immédiatement sans téléchargement

## Avantages

✅ **Pas de problèmes de permissions** - Vous utilisez votre propre dossier
✅ **Plus rapide** - Pas de vérification du cache système
✅ **Portable** - Vous pouvez copier le dossier sur une autre machine
✅ **Hors ligne** - Une fois téléchargé, fonctionne sans internet

## Structure des fichiers

```
P07_POC_LG/
├── Swin_Transformer              # Script principal
├── download_swin_model.py        # Script de téléchargement
├── swin_model_cache/             # Cache local (créé automatiquement)
│   └── models--microsoft--swin-tiny-patch4-window7-224/
└── README_CACHE.md               # Ce fichier
```

## En cas de problème

Si vous voyez encore l'erreur de cache système :
1. Vérifiez que `download_swin_model.py` s'est bien terminé avec succès
2. Vérifiez que le dossier `./swin_model_cache` existe et contient des fichiers
3. Relancez le téléchargement si nécessaire

## Suppression du cache

Pour supprimer le cache local (et libérer de l'espace disque) :

```bash
rm -rf ./swin_model_cache
```

Puis relancez `download_swin_model.py` si vous en avez à nouveau besoin.
