#!/usr/bin/env python3
"""
Script de test de fine-tuning avec données synthétiques
"""

import os
import sys
import json
import time
from pathlib import Path
import torch
import numpy as np
import librosa
import soundfile as sf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def create_synthetic_dataset():
    """Créer un petit dataset synthétique pour tester le training"""
    console.print("🎭 Création d'un dataset synthétique...", style="bold blue")
    
    # Vocabulaire ferroviaire français
    cities = ["Lyon", "Paris", "Bordeaux", "Chalon", "Chauny", "Nogent"]
    train_types = ["TER", "TGV", "Intercités"]
    actions = ["arrive", "part", "entre en gare", "dessert"]
    platforms = ["voie 1", "voie 2", "voie 3", "quai A", "quai B"]
    
    # Templates de phrases
    templates = [
        "Le {train_type} à destination de {city} {action}",
        "Le train pour {city} {action} {platform}",
        "Correspondance pour {city}",
        "Prochain arrêt {city}",
        "{train_type} {city} {platform}",
        "Attention le train à destination de {city} va partir"
    ]
    
    dataset_dir = Path("data/synthetic_train")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    manifests = {"train": [], "val": []}
    sample_rate = 16000
    
    console.print("🔄 Génération des échantillons...")
    
    # Générer les échantillons
    total_samples = 50  # Petit dataset pour test
    train_samples = int(total_samples * 0.8)
    
    for i in range(total_samples):
        # Choisir split
        split = "train" if i < train_samples else "val"
        
        # Générer phrase
        template = np.random.choice(templates)
        text = template.format(
            city=np.random.choice(cities),
            train_type=np.random.choice(train_types),
            action=np.random.choice(actions),
            platform=np.random.choice(platforms)
        )
        
        # Créer audio synthétique
        duration = len(text) * 0.08 + np.random.uniform(0.5, 1.5)
        num_samples = int(sample_rate * duration)
        
        # Signal complexe simulant la parole
        t = np.linspace(0, duration, num_samples)
        
        # Fréquences fondamentales variables
        f0 = 150 + 50 * np.sin(2 * np.pi * 0.5 * t)  # Intonation
        
        # Générer harmoniques
        signal = np.zeros(num_samples)
        for harmonic in range(1, 6):
            amplitude = 1.0 / harmonic
            signal += amplitude * np.sin(2 * np.pi * f0 * harmonic * t)
        
        # Modulation pour simuler syllables
        syllable_rate = len(text.split()) * 2 / duration  # ~2 syllabes par mot
        modulation = 1 + 0.4 * (np.sin(2 * np.pi * syllable_rate * t) > 0)
        signal *= modulation
        
        # Ajout de bruit léger
        noise = np.random.normal(0, 0.05, signal.shape)
        signal += noise
        
        # Normalisation
        signal = signal / np.max(np.abs(signal)) * 0.8
        
        # Fade in/out
        fade_samples = int(0.01 * sample_rate)  # 10ms
        signal[:fade_samples] *= np.linspace(0, 1, fade_samples)
        signal[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        # Sauvegarder
        audio_filename = f"synthetic_{split}_{i:03d}.wav"
        audio_path = dataset_dir / audio_filename
        sf.write(audio_path, signal, sample_rate)
        
        # Ajouter au manifest
        manifests[split].append({
            "audio_filepath": str(audio_path.absolute()),
            "text": text,
            "duration": duration
        })
    
    # Sauvegarder les manifests
    for split, manifest in manifests.items():
        manifest_path = Path("data/manifests") / f"{split}_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            for item in manifest:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        console.print(f"✅ {split}: {len(manifest)} échantillons -> {manifest_path}")
    
    return manifests

def create_training_config():
    """Créer une configuration de training pour test"""
    console.print("⚙️ Création de la configuration de training...", style="bold blue")
    
    config = {
        "name": "ConformerRNNT_Test",
        "model": {
            "sample_rate": 16000,
            "train_ds": {
                "manifest_filepath": "data/manifests/train_manifest.json",
                "batch_size": 2,
                "shuffle": True,
                "num_workers": 2,
                "pin_memory": True,
                "max_duration": 20.0,
                "min_duration": 0.1
            },
            "validation_ds": {
                "manifest_filepath": "data/manifests/val_manifest.json", 
                "batch_size": 2,
                "shuffle": False,
                "num_workers": 2,
                "pin_memory": True
            },
            "optim": {
                "name": "adamw",
                "lr": 1e-4,
                "betas": [0.9, 0.98],
                "weight_decay": 1e-3,
                "sched": {
                    "name": "CosineAnnealing",
                    "warmup_steps": 100,
                    "warmup_ratio": None,
                    "min_lr": 1e-6
                }
            }
        },
        "trainer": {
            "devices": 1,
            "max_epochs": 3,
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "strategy": "auto",
            "log_every_n_steps": 5,
            "val_check_interval": 20,
            "check_val_every_n_epoch": 1,
            "enable_checkpointing": True,
            "logger": False,  # Désactiver pour test
            "enable_model_summary": True
        }
    }
    
    config_path = Path("config/test_training_config.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder en YAML
    import yaml
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    console.print(f"✅ Configuration sauvegardée: {config_path}")
    return config_path

def load_model_for_training():
    """Charger le modèle pré-entraîné pour le fine-tuning"""
    console.print("🤖 Chargement du modèle pour training...", style="bold blue")
    
    try:
        import nemo.collections.asr as nemo_asr
        
        # Chercher le modèle pré-entraîné
        model_dir = Path("models/pretrained")
        nemo_files = list(model_dir.glob("*.nemo"))
        
        if not nemo_files:
            console.print("❌ Modèle pré-entraîné non trouvé")
            return None
        
        model_path = nemo_files[0]
        console.print(f"📂 Chargement depuis: {model_path}")
        
        # Charger le modèle
        model = nemo_asr.models.ASRModel.restore_from(str(model_path))
        
        # Configuration pour fine-tuning
        model.train()
        
        console.print("✅ Modèle chargé en mode training")
        return model
        
    except Exception as e:
        console.print(f"❌ Erreur chargement: {e}")
        return None

def test_data_loading(model, config_path):
    """Tester le chargement des données"""
    console.print("📊 Test du chargement de données...", style="bold blue")
    
    try:
        from omegaconf import OmegaConf
        
        # Charger la configuration
        cfg = OmegaConf.load(config_path)
        
        # Configurer les datasets
        model.setup_training_data(cfg.model.train_ds)
        model.setup_validation_data(cfg.model.validation_ds)
        
        # Tester un batch
        train_loader = model._train_dl
        val_loader = model._validation_dl
        
        console.print(f"✅ Train loader: {len(train_loader)} batches")
        console.print(f"✅ Val loader: {len(val_loader)} batches")
        
        # Tester le premier batch
        train_batch = next(iter(train_loader))
        console.print(f"✅ Batch shape: {train_batch[0].shape}")
        console.print(f"✅ Batch audio length: {train_batch[1].shape}")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Erreur chargement données: {e}")
        return False

def test_forward_pass(model):
    """Tester le forward pass du modèle"""
    console.print("🔄 Test du forward pass...", style="bold blue")
    
    try:
        # Créer un batch factice
        batch_size = 2
        max_seq_len = 1000
        feat_dim = 80  # Dimension des features mel-spectrogram
        
        # Audio features (mel-spectrogram)
        audio_signal = torch.randn(batch_size, max_seq_len, feat_dim)
        audio_lengths = torch.tensor([max_seq_len, max_seq_len // 2])
        
        # Targets (tokenized text)
        max_target_len = 50
        targets = torch.randint(0, model.decoder.vocabulary_size, (batch_size, max_target_len))
        target_lengths = torch.tensor([max_target_len, max_target_len // 2])
        
        # Forward pass
        with torch.no_grad():
            loss, num_targets, prediction = model.forward(
                input_signal=audio_signal,
                input_signal_length=audio_lengths,
                targets=targets,
                target_length=target_lengths
            )
        
        console.print(f"✅ Loss: {loss.item():.4f}")
        console.print(f"✅ Num targets: {num_targets}")
        console.print(f"✅ Forward pass réussi")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Erreur forward pass: {e}")
        return False

def run_mini_training(model, config_path):
    """Lancer un mini training de test"""
    console.print("🏃 Lancement du mini training...", style="bold blue")
    
    try:
        import pytorch_lightning as pl
        from omegaconf import OmegaConf
        
        # Charger la configuration
        cfg = OmegaConf.load(config_path)
        
        # Configurer le modèle
        model.setup_training_data(cfg.model.train_ds)
        model.setup_validation_data(cfg.model.validation_ds)
        model.setup_optimization(cfg.model.optim)
        
        # Créer le trainer
        trainer = pl.Trainer(
            max_epochs=1,  # Juste 1 epoch pour test
            devices=1,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            log_every_n_steps=1,
            val_check_interval=5,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=True
        )
        
        console.print("🔄 Démarrage du training test...")
        
        # Sauvegarder la loss initiale
        model.eval()
        with torch.no_grad():
            val_batch = next(iter(model._validation_dl))
            initial_loss = model.validation_step(val_batch, 0)
        
        console.print(f"📊 Loss initiale: {initial_loss:.4f}")
        
        # Training
        start_time = time.time()
        trainer.fit(model)
        training_time = time.time() - start_time
        
        # Vérifier la loss finale
        model.eval()
        with torch.no_grad():
            final_loss = model.validation_step(val_batch, 0)
        
        console.print(f"📊 Loss finale: {final_loss:.4f}")
        console.print(f"📊 Différence: {initial_loss - final_loss:.4f}")
        console.print(f"⏱️ Temps training: {training_time:.2f}s")
        
        # Sauvegarder le modèle fine-tuné
        checkpoint_dir = Path("models/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / "test_finetuned.nemo"
        model.save_to(str(checkpoint_path))
        
        console.print(f"✅ Modèle sauvegardé: {checkpoint_path}")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Erreur training: {e}")
        import traceback
        console.print(traceback.format_exc())
        return False

def test_inference_after_training():
    """Tester l'inférence après fine-tuning"""
    console.print("🎯 Test d'inférence post-training...", style="bold blue")
    
    try:
        import nemo.collections.asr as nemo_asr
        
        # Charger le modèle fine-tuné
        checkpoint_path = Path("models/checkpoints/test_finetuned.nemo")
        
        if not checkpoint_path.exists():
            console.print("❌ Modèle fine-tuné non trouvé")
            return False
        
        model = nemo_asr.models.ASRModel.restore_from(str(checkpoint_path))
        model.eval()
        
        # Test avec phrase ferroviaire
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Signal test avec patterns plus complexes
        signal = (np.sin(2 * np.pi * 200 * t) + 
                 0.5 * np.sin(2 * np.pi * 400 * t) +
                 0.3 * np.sin(2 * np.pi * 600 * t))
        
        # Modulation pour simuler "Lyon"
        signal *= (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
        
        # Transcription
        transcription = model.transcribe([signal])
        
        console.print(f"✅ Transcription test: '{transcription[0]}'")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Erreur test inférence: {e}")
        return False

def main():
    """Fonction principale"""
    console.print("🧪 Test de Fine-tuning Conformer RNN-T", style="bold")
    console.print("=" * 60)
    
    # Étapes du test
    steps = [
        ("Création dataset synthétique", create_synthetic_dataset),
        ("Configuration training", create_training_config),
        ("Chargement modèle", load_model_for_training),
    ]
    
    results = {}
    model = None
    config_path = None
    
    # Exécuter les étapes préparatoires
    for step_name, step_func in steps:
        console.print(f"\n🔄 {step_name}...")
        try:
            if step_name == "Création dataset synthétique":
                results[step_name] = step_func() is not None
            elif step_name == "Configuration training":
                config_path = step_func()
                results[step_name] = config_path is not None
            elif step_name == "Chargement modèle":
                model = step_func()
                results[step_name] = model is not None
                
        except Exception as e:
            console.print(f"❌ Erreur {step_name}: {e}")
            results[step_name] = False
    
    # Tests avec le modèle
    if model and config_path:
        training_steps = [
            ("Test chargement données", lambda: test_data_loading(model, config_path)),
            ("Test forward pass", lambda: test_forward_pass(model)),
            ("Mini training", lambda: run_mini_training(model, config_path)),
            ("Test inférence post-training", test_inference_after_training)
        ]
        
        for step_name, step_func in training_steps:
            console.print(f"\n🔄 {step_name}...")
            try:
                results[step_name] = step_func()
            except Exception as e:
                console.print(f"❌ Erreur {step_name}: {e}")
                results[step_name] = False
    
    # Résumé
    console.print("\n📋 Résumé du test de training:", style="bold")
    
    for step_name, success in results.items():
        status = "✅" if success else "❌"
        console.print(f"  {status} {step_name}")
    
    success_count = sum(results.values())
    total_steps = len(results)
    
    console.print(f"\n🎯 Score: {success_count}/{total_steps} étapes réussies")
    
    if success_count == total_steps:
        console.print("🎉 Pipeline de training fonctionnel!", style="bold green")
        console.print("\n📝 Prochaines étapes:")
        console.print("1. Préparer vos vraies données audio")
        console.print("2. Ajuster la configuration pour votre dataset")
        console.print("3. Lancer le fine-tuning complet")
    else:
        console.print("⚠️ Certaines étapes ont échoué.", style="bold yellow")
        console.print("Vérifiez les logs pour diagnostiquer les problèmes.")
    
    return success_count == total_steps

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)