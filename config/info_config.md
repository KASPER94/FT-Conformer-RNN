Le model_config.yaml définit TOUTE l'architecture et les hyperparamètres de votre modèle Conformer RNN-T. C'est le fichier qui dit à NeMo :

Comment construire le modèle
Quels paramètres utiliser
Comment traiter les données

🏗️ Sections Clés Expliquées
1. Architecture du Modèle
yamlencoder:
  n_layers: 16        # Nombre de couches Conformer
  d_model: 512        # Dimension des embeddings
  n_heads: 8          # Têtes d'attention
→ Définit la "taille" et complexité du modèle
2. Préprocessing Audio
yamlpreprocessor:
  sample_rate: 16000
  features: 80        # Mel-spectrogramme 80 dimensions
  window_size: 0.025  # Fenêtre de 25ms
→ Comment transformer l'audio en features
3. Configuration d'Entraînement
yamltrain_ds:
  batch_size: 8
  manifest_filepath: "data/manifests/train_manifest.json"
  max_duration: 20.0
→ Où trouver les données et comment les charger
4. Optimisation
yamloptim:
  lr: 1e-4           # Learning rate
  name: adamw        # Optimiseur
→ Comment le modèle apprend
🔄 Utilisation Pratique
Lors du fine-tuning :
python# NeMo lit ce fichier pour construire le modèle
from omegaconf import OmegaConf
cfg = OmegaConf.load("config/model_config.yaml")

# Le modèle est créé selon cette config
model = nemo_asr.models.EncDecRNNTModel(cfg=cfg.model)
Lors de l'inférence :
python# La config est sauvée DANS le modèle .nemo
model = nemo_asr.models.ASRModel.restore_from("model.nemo")
# Il "se souvient" de sa configuration originale
⚙️ Avantages de cette Approche
1. Reproductibilité
yaml# Même config = même modèle
# Partage facile entre équipes
seed: 42
2. Expérimentation Facile
yaml# Tester différentes tailles
encoder:
  n_layers: 12  # vs 16 vs 24
  d_model: 256  # vs 512 vs 1024
3. Adaptation Domaine
yaml# Section spéciale pour votre cas ferroviaire
railway_config:
  cities: ["lyon", "bordeaux", "chalon"]
  vocabulary_weights:
    cities: 2.0  # Pondérer les noms de villes
🎛️ Paramètres Critiques à Ajuster
Pour Performance :
yamlencoder:
  n_layers: 16      # Plus = meilleur mais plus lent
  d_model: 512      # Dimension des représentations
Pour Vitesse :
yamltrain_ds:
  batch_size: 16    # Plus grand = plus rapide (si GPU le permet)
  num_workers: 8    # Parallélisation du loading
Pour Qualité Audio :
yamlpreprocessor:
  features: 80      # Plus = plus d'info audio
  window_size: 0.025 # Résolution temporelle
🔧 Modification en Cours de Projet
Vous pouvez créer plusieurs configs :
bashconfig/
├── model_config.yaml        # Config de base
├── model_config_small.yaml  # Version rapide
├── model_config_large.yaml  # Version qualité max
└── model_config_inference.yaml # Version déploiement
Puis les utiliser :
python# Pour training rapide
model = load_model("config/model_config_small.yaml")

# Pour production
model = load_model("config/model_config_large.yaml")
🎯 Cas d'Usage Concrets
Debug : Modèle trop lent
yaml# Dans model_config.yaml, réduire :
encoder:
  n_layers: 8    # au lieu de 16
  d_model: 256   # au lieu de 512
train_ds:
  batch_size: 4  # au lieu de 8
Production : Besoin de qualité max
yaml# Augmenter :
encoder:
  n_layers: 24
  d_model: 1024
  n_heads: 16
Fine-tuning : Focus sur vos données
yaml# Ajuster pour votre domaine :
train_ds:
  manifest_filepath: "data/manifests/railway_train.json"
  max_duration: 15.0  # Vos annonces sont courtes
spec_augment:
  freq_masks: 3       # Plus d'augmentation
🔍 En Résumé
Le model_config.yaml est votre tableau de bord central :

✅ Architecture : Définit la structure du modèle
✅ Données : Où et comment charger vos données
✅ Training : Tous les hyperparamètres d'entraînement
✅ Reproductibilité : Même config = même résultats
✅ Expérimentation : Facile de tester des variantes

Sans ce fichier, NeMo ne sait pas :

Combien de couches mettre dans l'encodeur
Quelle taille de batch utiliser
Où trouver vos données d'entraînement
Comment optimiser le modèle