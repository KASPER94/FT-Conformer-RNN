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
python scripts/01_setup_environment.py
```

### 2. Téléchargement du modèle pré-entraîné

```bash
python scripts/02_download_model.py
```

### 3. Tests de fonctionnement

```bash
# Test d'inférence
python scripts/03_test_inference.py

# Test de fine-tuning
python scripts/05_test_training.py
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
"Attention à Nogent-le-Rotrou, terminus"
```

### Vocabulaire Cible

**Villes françaises** : Lyon, Paris, Bordeaux, Chalon-sur-Saône, Chauny, Nogent-le-Rotrou, Roanne, Saint-Étienne...

**Terminologie ferroviaire** : TER, TGV, Intercités, correspondance, terminus, quai, voie, desserte, omnibus...

## 🎛️ Configuration

### Paramètres de Training

```yaml
model:
  name: "stt_conformer_rnnt_large"
  sample_rate: 16000
  
training:
  batch_size: 8-16
  learning_rate: 1e-4 à 5e-5
  max_epochs: 10-50
  
data:
  max_duration: 20.0
  min_duration: 0.5
  augmentation: true
```

### Fine-tuning Strategy

1. **Phase 1** : Gel des couches basses, fine-tuning des couches hautes
2. **Phase 2** : Fine-tuning complet avec learning rate réduit
3. **Phase 3** : Optimisation spécifique au domaine

## 🔧 Utilisation

### Inférence Simple

```python
import nemo.collections.asr as nemo_asr

# Charger le modèle fine-tuné
model = nemo_asr.models.ASRModel.restore_from("models/final/railway_model.nemo")

# Transcription
transcription = model.transcribe(["path/to/audio.wav"])
print(transcription[0])
```

### Streaming en Temps Réel

```python
from src.inference.streaming import StreamingASR

# Initialiser le système streaming
asr = StreamingASR("models/final/railway_model.nemo")

# Traitement en chunks
for audio_chunk in audio_stream:
    partial_transcript = asr.process_chunk(audio_chunk)
    print(f"Transcription partielle: {partial_transcript}")
```

## 📈 Performance Attendue

### Métriques

- **WER (Word Error Rate)** : < 5% sur données ferroviaires
- **Latence** : < 200ms pour streaming
- **RTF (Real-Time Factor)** : < 0.3 sur GPU moderne
- **Couverture vocabulaire** : > 95% termes ferroviaires français

### Benchmarks

| Durée Audio | Temps Process | RTF | Performance |
|-------------|---------------|-----|-------------|
| 1.0s        | 0.1s         | 0.1 | 🚀 Excellent |
| 5.0s        | 0.8s         | 0.16| ✅ Bon       |
| 10.0s       | 2.1s         | 0.21| ✅ Bon       |

## 🔄 Pipeline de Développement

### 1. Préparation des Données

```bash
# Préparer le dataset
python scripts/04_prepare_data.py --input_dir data/raw --output_dir data/processed

# Vérifier la qualité
python scripts/06_evaluate_model.py --mode data_quality
```

### 2. Fine-tuning

```bash
# Lancer l'entraînement
python -m src.training.trainer --config config/training_config.yaml

# Monitoring
tensorboard --logdir logs/tensorboard
```

### 3. Évaluation

```bash
# Test sur données de validation
python scripts/06_evaluate_model.py --model_path models/checkpoints/best.nemo

# Test streaming
python scripts/03_test_inference.py --streaming
```

## 🎭 Voice Cloning (Bonus)

Le dataset peut être réutilisé pour créer des voix synthétiques :

### Prérequis Voice Cloning

- **Locuteur principal** : 60-90 minutes d'audio
- **Qualité constante** : même microphone, environnement
- **Diversité prosodique** : questions, affirmations, urgence

### Pipeline TTS

```python
# Après fine-tuning ASR, utiliser pour TTS
from src.models.voice_cloning import RailwayTTS

tts = RailwayTTS("models/voice_clone/speaker_001.nemo")
audio = tts.synthesize("Le train à destination de Lyon va partir")
```

## 🛠️ Scripts Disponibles

| Script | Description |
|--------|-------------|
| `01_setup_environment.py` | Configuration initiale |
| `02_download_model.py` | Téléchargement modèle pré-entraîné |
| `03_test_inference.py` | Tests d'inférence et streaming |
| `04_prepare_data.py` | Préparation du dataset |
| `05_test_training.py` | Test de fine-tuning |
| `06_evaluate_model.py` | Évaluation et métriques |

## 📋 Checklist de Déploiement

### Avant Production

- [ ] Dataset de 8+ heures validé
- [ ] Fine-tuning convergé (WER < 5%)
- [ ] Tests streaming réussis (latence < 200ms)
- [ ] Vocabulaire ferroviaire couvert (> 95%)
- [ ] Tests en environnement bruité
- [ ] Validation sur voix non vues

### Optimisations Production

- [ ] Quantization du modèle (FP16/INT8)
- [ ] Optimisation TensorRT
- [ ] Cache des modèles
- [ ] Monitoring temps réel
- [ ] Fallback en cas d'échec

## 🔍 Troubleshooting

### Problèmes Courants

**CUDA Out of Memory**
```bash
# Réduire batch_size dans config
batch_size: 4  # au lieu de 8
```

**Convergence lente**
```bash
# Ajuster learning rate
learning_rate: 5e-5  # au lieu de 1e-4
```

**Mauvaise reconnaissance des villes**
```bash
# Augmenter données spécifiques
python scripts/04_prepare_data.py --focus_cities Lyon,Bordeaux,Chalon
```

## 📚 Ressources

### Documentation

- [NVIDIA NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)
- [Conformer Paper](https://arxiv.org/abs/2005.08100)
- [RNN-T Architecture](https://arxiv.org/abs/1211.3711)

### Modèles Pré-entraînés

- `stt_conformer_rnnt_small` : Modèle léger, rapide
- `stt_conformer_rnnt_medium` : Bon compromis performance/vitesse
- `stt_conformer_rnnt_large` : Meilleure qualité, plus lent

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/amelioration`)
3. Commit les changements (`git commit -am 'Ajout fonctionnalité'`)
4. Push vers la branche (`git push origin feature/amelioration`)
5. Créer une Pull Request

## 📄 License

Ce projet est sous licence MIT. Voir `LICENSE` pour plus de détails.

## 🙏 Remerciements

- **NVIDIA** pour les modèles NeMo pré-entraînés
- **Équipe Conformer** pour l'architecture innovante
- **Communauté ASR** pour les contributions open source

---

**🚀 Prêt à commencer ?**

```bash
python scripts/01_setup_environment.py
```

Pour toute question : [Issues GitHub](link-to-issues) | [Documentation](link-to-docs)