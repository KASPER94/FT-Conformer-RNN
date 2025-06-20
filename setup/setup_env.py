#!/usr/bin/env python3
"""
Script de configuration de l'environnement pour le projet Conformer RNN-T
"""

import os
import sys
import subprocess
import pkg_resources
from pathlib import Path
import torch

def check_cuda():
    """Vérifier la disponibilité de CUDA"""
    print("🔍 Vérification de CUDA...")
    if torch.cuda.is_available():
        print(f"✅ CUDA disponible - Version: {torch.version.cuda}")
        print(f"✅ GPU détecté: {torch.cuda.get_device_name(0)}")
        print(f"✅ Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("⚠️  CUDA non disponible - Le training sera plus lent sur CPU")
        return False

def check_dependencies():
    """Vérifier les dépendances installées"""
    print("\n🔍 Vérification des dépendances...")
    
    required_packages = [
        'torch',
        'torchaudio', 
        'nemo_toolkit',
        'librosa',
        'soundfile',
        'omegaconf'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            pkg_resources.get_distribution(package)
            print(f"✅ {package} installé")
        except pkg_resources.DistributionNotFound:
            print(f"❌ {package} manquant")
            missing_packages.append(package)
    
    return missing_packages

def install_missing_packages(missing_packages):
    """Installer les packages manquants"""
    if missing_packages:
        print(f"\n📦 Installation des packages manquants: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'] + missing_packages
            )
            print("✅ Installation terminée")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur lors de l'installation: {e}")
            return False
    return True

def create_directories():
    """Créer la structure de dossiers"""
    print("\n📁 Création de la structure de dossiers...")
    
    directories = [
        'data/raw/audio',
        'data/raw/transcripts', 
        'data/processed/train',
        'data/processed/val',
        'data/processed/test',
        'data/manifests',
        'models/pretrained',
        'models/checkpoints',
        'models/final',
        'logs/tensorboard',
        'logs/training'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")

def test_nemo_import():
    """Tester l'import de NeMo"""
    print("\n🧪 Test d'import de NeMo...")
    try:
        import nemo
        import nemo.collections.asr as nemo_asr
        print(f"✅ NeMo version: {nemo.__version__}")
        print("✅ NeMo ASR importé avec succès")
        return True
    except ImportError as e:
        print(f"❌ Erreur d'import NeMo: {e}")
        return False

def test_audio_processing():
    """Tester les outils de traitement audio"""
    print("\n🎵 Test des outils audio...")
    try:
        import librosa
        import soundfile as sf
        import torch
        import torchaudio
        
        print(f"✅ librosa version: {librosa.__version__}")
        print(f"✅ soundfile importé")
        print(f"✅ torchaudio version: {torchaudio.__version__}")
        
        # Test création d'un signal audio simple
        sample_rate = 16000
        duration = 1.0
        t = torch.linspace(0, duration, int(sample_rate * duration))
        signal = torch.sin(2 * 3.14159 * 440 * t)  # 440 Hz sine wave
        
        print("✅ Génération de signal audio test réussie")
        return True
        
    except Exception as e:
        print(f"❌ Erreur test audio: {e}")
        return False

def create_sample_config():
    """Créer un fichier de configuration d'exemple"""
    print("\n⚙️  Création de la configuration d'exemple...")
    
    config_content = """
# Configuration d'exemple pour Conformer RNN-T
model:
  name: "stt_conformer_rnnt_large"
  pretrained: true
  
training:
  batch_size: 8
  learning_rate: 1e-4
  max_epochs: 10
  
data:
  sample_rate: 16000
  max_duration: 20.0
  min_duration: 0.5
  
augmentation:
  noise_prob: 0.3
  speed_perturb: true
  
logging:
  log_every_n_steps: 100
  val_check_interval: 1000
"""
    
    config_path = Path("config/test_config.yaml")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content.strip())
    
    print(f"✅ Configuration créée: {config_path}")

def main():
    """Fonction principale"""
    print("🚀 Configuration de l'environnement Conformer RNN-T")
    print("=" * 50)
    
    # Vérifications système
    cuda_available = check_cuda()
    missing_packages = check_dependencies()
    
    # Installation si nécessaire
    if missing_packages:
        if not install_missing_packages(missing_packages):
            print("❌ Échec de l'installation. Vérifiez manuellement.")
            return False
    
    # Création de la structure
    create_directories()
    
    # Tests fonctionnels
    if not test_nemo_import():
        print("❌ NeMo non fonctionnel")
        return False
        
    if not test_audio_processing():
        print("❌ Outils audio non fonctionnels")
        return False
    
    # Configuration
    create_sample_config()
    
    print("\n🎉 Configuration terminée avec succès!")
    print("\nÉtapes suivantes:")
    print("1. Exécuter: python scripts/02_download_model.py")
    print("2. Tester l'inférence: python scripts/03_test_inference.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)