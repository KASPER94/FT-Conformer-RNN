#!/usr/bin/env python3
"""
Script de téléchargement du modèle Conformer RNN-T pré-entraîné
"""

import os
import sys
from pathlib import Path
import torch
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn

console = Console()

def download_pretrained_model():
    """Télécharger le modèle pré-entraîné NVIDIA"""
    console.print("📥 Téléchargement du modèle Conformer RNN-T...", style="bold blue")
    
    try:
        import nemo.collections.asr as nemo_asr
        
        # Modèles disponibles (du plus petit au plus grand)
        models = {
            "small": "stt_conformer_rnnt_small",
            "medium": "stt_conformer_rnnt_medium", 
            "large": "stt_conformer_rnnt_large"
        }
        
        console.print("Modèles disponibles:")
        for size, model_name in models.items():
            console.print(f"  - {size}: {model_name}")
        
        # Commencer par le modèle medium pour les tests
        model_name = models["medium"]
        console.print(f"\n🔄 Téléchargement de {model_name}...")
        
        # Télécharger le modèle
        model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        
        # Sauvegarder localement
        model_path = Path("models/pretrained")
        model_path.mkdir(parents=True, exist_ok=True)
        
        local_path = model_path / f"{model_name}.nemo"
        model.save_to(str(local_path))
        
        console.print(f"✅ Modèle sauvegardé: {local_path}", style="bold green")
        
        # Afficher les informations du modèle
        console.print("\n📊 Informations du modèle:")
        console.print(f"  - Architecture: {model.__class__.__name__}")
        console.print(f"  - Vocabulaire: {model.decoder.vocabulary}")
        console.print(f"  - Sample rate: {model._cfg.sample_rate}Hz")
        
        return model, str(local_path)
        
    except Exception as e:
        console.print(f"❌ Erreur lors du téléchargement: {e}", style="bold red")
        return None, None

def verify_model_components(model):
    """Vérifier les composants du modèle"""
    console.print("\n🔍 Vérification des composants du modèle...")
    
    try:
        # Vérifier l'encodeur
        if hasattr(model, 'encoder'):
            console.print("✅ Encodeur Conformer présent")
        else:
            console.print("❌ Encodeur manquant")
            
        # Vérifier le décodeur RNN-T
        if hasattr(model, 'decoder'):
            console.print("✅ Décodeur RNN-T présent")
        else:
            console.print("❌ Décodeur manquant")
            
        # Vérifier le joint network
        if hasattr(model, 'joint'):
            console.print("✅ Joint Network présent")
        else:
            console.print("❌ Joint Network manquant")
            
        return True
        
    except Exception as e:
        console.print(f"❌ Erreur lors de la vérification: {e}")
        return False

def test_model_inference(model):
    """Tester l'inférence avec un signal audio factice"""
    console.print("\n🧪 Test d'inférence avec signal audio factice...")
    
    try:
        # Créer un signal audio factice (16kHz, 2 secondes)
        sample_rate = 16000
        duration = 2.0
        num_samples = int(sample_rate * duration)
        
        # Signal sinusoïdal simple
        import torch
        import numpy as np
        
        t = np.linspace(0, duration, num_samples)
        # Mélange de fréquences pour simuler de la parole
        signal = (np.sin(2 * np.pi * 200 * t) + 
                 0.5 * np.sin(2 * np.pi * 400 * t) + 
                 0.3 * np.sin(2 * np.pi * 800 * t))
        
        # Ajouter un peu de bruit
        noise = np.random.normal(0, 0.1, signal.shape)
        signal = signal + noise
        
        # Normaliser
        signal = signal / np.max(np.abs(signal))
        
        console.print(f"✅ Signal audio créé: {duration}s, {sample_rate}Hz")
        
        # Test d'inférence
        with torch.no_grad():
            transcription = model.transcribe([signal])
            
        console.print(f"✅ Inférence réussie")
        console.print(f"  - Transcription (signal factice): '{transcription[0]}'")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Erreur lors du test d'inférence: {e}")
        return False

def save_model_info(model, model_path):
    """Sauvegarder les informations du modèle"""
    console.print("\n💾 Sauvegarde des informations du modèle...")
    
    try:
        import json
        
        model_info = {
            "model_name": model.__class__.__name__,
            "model_path": model_path,
            "vocabulary": list(model.decoder.vocabulary),
            "sample_rate": model._cfg.sample_rate,
            "config": model._cfg if hasattr(model, '_cfg') else {}
        }
        
        info_path = Path("models/pretrained/model_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
            
        console.print(f"✅ Informations sauvegardées: {info_path}")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Erreur sauvegarde info: {e}")
        return False

def main():
    """Fonction principale"""
    console.print("🤖 Téléchargement et test du modèle Conformer RNN-T", style="bold")
    console.print("=" * 60)
    
    # Télécharger le modèle
    model, model_path = download_pretrained_model()
    if model is None:
        console.print("❌ Échec du téléchargement", style="bold red")
        return False
    
    # Vérifier les composants
    if not verify_model_components(model):
        console.print("❌ Composants du modèle invalides", style="bold red")
        return False
    
    # Tester l'inférence
    if not test_model_inference(model):
        console.print("❌ Test d'inférence échoué", style="bold red")
        return False
    
    # Sauvegarder les infos
    save_model_info(model, model_path)
    
    console.print("\n🎉 Modèle téléchargé et testé avec succès!", style="bold green")
    console.print("\nÉtapes suivantes:")
    console.print("1. Tester avec de l'audio réel: python scripts/03_test_inference.py")
    console.print("2. Préparer vos données: python scripts/04_prepare_data.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)