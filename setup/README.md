# 🚂 Projet ASR Ferroviaire - Conformer RNN-T

Ce projet implémente un système de reconnaissance vocale automatique (ASR) basé sur l'architecture **Conformer RNN-T** de NVIDIA, spécialement fine-tuné pour le domaine ferroviaire français.

## 🎯 Objectifs

- **Reconnaissance vocale temps réel** pour annonces ferroviaires
- **Vocabulaire spécialisé** : noms de villes françaises, terminologie SNCF
- **Streaming capability** : latence faible pour applications en direct
- **Robustesse** : fonctionnement dans environnement bruité (gares, trains)

## 🏗️ Architecture

### Pourquoi Conformer RNN-T ?

**Conformer** combine les avantages de :
- **Transformers** : Attention globale pour capturer les dépendances long-terme
- **CNN** : Patterns locaux pour les caractéristiques phonétiques
- **Architecture hybride** optimale pour la reconnaissance vocale

**RNN-Transducer** permet :
- **Streaming natif** : décodage incrémental pendant que l'audio arrive
- **Latence faible** : pas d'attente de fin d'énoncé
- **Gestion intelligente des silences** via le mécanisme de "blank token"

### Structure du Projet

```
railway_asr_project/
├── src/                    # Code source principal
│   ├── models/            # Modèles et architectures
│   ├── data/              # Gestion des données
│   ├── training/          # Pipeline d'entraînement
│   └── inference/         # Système d'inférence
├── scripts/               # Scripts d'installation et test
├── config/                # Configurations YAML
├── data/                  # Données d'entraînement
└── models/                # Modèles sauvegardés
```

## 🚀 Installation Rapide

### 1. Configuration de l'environnement

```bash
# Cloner le projet
git clone <repo-url>
cd railway_asr_project

# Installer les dépendances
pip install -r requirements.txt

# Configurer l'environnement
python setup/setup_env.py
```

### 2. Téléchargement du modèle pré-entraîné

```bash
python setup/download_model.py
```

### 3. Tests de fonctionnement

```bash
# Test d'inférence
python setup/test_inference.py

# Test de fine-tuning
python setup/test_training.py
```

## 📊 Dataset Recommandé

### Spécifications

- **Volume minimum** : 8-12 heures d'audio
- **Locuteurs** : 15-25 personnes (hommes/femmes, âges variés)
- **Qualité audio** : 16kHz, mono, WAV
- **Contenu** : Phrases ferroviaires avec noms de villes françaises

### Exemples de Phrases

```
"Le TER à destination de Lyon partira voie 3"
"Correspondance pour Chalon-sur-Saône"
"Prochain arrêt Chauny"
"Le train pour Bordeaux entre en gare"
"Attention à Nogent-le-