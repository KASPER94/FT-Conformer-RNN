#!/usr/bin/env python3
"""
Script de test d'inférence avec audio réel et streaming
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
import torch
import librosa
import soundfile as sf
from rich.console import Console
from rich.table import Table

console = Console()

def load_model():
    """Charger le modèle téléchargé"""
    console.print("🔄 Chargement du modèle...", style="bold blue")
    
    try:
        import nemo.collections.asr as nemo_asr
        
        # Chercher le modèle local
        model_dir = Path("models/pretrained")
        nemo_files = list(model_dir.glob("*.nemo"))
        
        if not nemo_files:
            console.print("❌ Aucun modèle trouvé. Exécutez d'abord 02_download_model.py")
            return None
            
        model_path = nemo_files[0]
        console.print(f"📂 Chargement depuis: {model_path}")
        
        model = nemo_asr.models.ASRModel.restore_from(str(model_path))
        model.eval()
        
        console.print("✅ Modèle chargé avec succès", style="bold green")
        return model
        
    except Exception as e:
        console.print(f"❌ Erreur chargement modèle: {e}", style="bold red")
        return None

def create_test_audio():
    """Créer des fichiers audio de test pour les noms de villes"""
    console.print("\n🎵 Création d'audio de test avec TTS...", style="bold blue")
    
    # Phrases de test avec noms de villes françaises
    test_phrases = [
        "Le train à destination de Lyon partira voie trois",
        "Correspondance pour Chalon-sur-Saône", 
        "Prochain arrêt Chauny",
        "Le TER pour Bordeaux entre en gare",
        "Attention à Nogent-le-Rotrou, terminus"
    ]
    
    test_dir = Path("data/test_audio")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Utiliser pyttsx3 pour créer des fichiers audio de test
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)  # Vitesse de parole
            
            created_files = []
            for i, phrase in enumerate(test_phrases):
                filename = test_dir / f"test_{i:02d}.wav"
                engine.save_to_file(phrase, str(filename))
                created_files.append((filename, phrase))
                
            engine.runAndWait()
            console.print(f"✅ {len(created_files)} fichiers audio créés")
            return created_files
            
        except ImportError:
            console.print("⚠️  pyttsx3 non installé, création d'audio synthétique...")
            return create_synthetic_audio(test_phrases, test_dir)
            
    except Exception as e:
        console.print(f"❌ Erreur création audio: {e}")
        return []

def create_synthetic_audio(phrases, test_dir):
    """Créer de l'audio synthétique simple"""
    created_files = []
    sample_rate = 16000
    
    for i, phrase in enumerate(phrases):
        # Audio blanc avec variation de fréquence
        duration = len(phrase) * 0.1 + 1.0  # Durée basée sur longueur du texte
        num_samples = int(sample_rate * duration)
        
        # Signal complexe simulant la parole
        t = np.linspace(0, duration, num_samples)
        signal = np.zeros(num_samples)
        
        # Ajouter plusieurs harmoniques
        for freq in [150, 300, 450, 600]:
            amplitude = 1.0 / (freq / 150)  # Amplitude décroissante
            signal += amplitude * np.sin(2 * np.pi * freq * t)
        
        # Modulation d'amplitude pour simuler les syllabes
        modulation = 1 + 0.3 * np.sin(2 * np.pi * 2 * t)
        signal *= modulation
        
        # Normaliser
        signal = signal / np.max(np.abs(signal)) * 0.7
        
        # Sauvegarder
        filename = test_dir / f"synthetic_{i:02d}.wav"
        sf.write(filename, signal, sample_rate)
        created_files.append((filename, phrase))
    
    console.print(f"✅ {len(created_files)} fichiers synthétiques créés")
    return created_files

def test_batch_inference(model, audio_files):
    """Tester l'inférence en lot"""
    console.print("\n🔍 Test d'inférence en lot...", style="bold blue")
    
    if not audio_files:
        console.print("❌ Aucun fichier audio disponible")
        return False
    
    try:
        # Préparer les chemins audio
        audio_paths = [str(path) for path, _ in audio_files]
        expected_texts = [text for _, text in audio_files]
        
        start_time = time.time()
        transcriptions = model.transcribe(audio_paths)
        inference_time = time.time() - start_time
        
        # Créer un tableau des résultats
        table = Table(title="Résultats d'inférence")
        table.add_column("Fichier", style="cyan")
        table.add_column("Texte attendu", style="green")
        table.add_column("Transcription", style="yellow")
        table.add_column("Similaire", style="magenta")
        
        for i, (audio_path, expected, transcription) in enumerate(zip(audio_paths, expected_texts, transcriptions)):
            filename = Path(audio_path).name
            # Simple vérification de similarité
            similar = "🟢" if any(word.lower() in transcription.lower() for word in expected.split() if len(word) > 3) else "🔴"
            table.add_row(filename, expected, transcription, similar)
        
        console.print(table)
        console.print(f"\n⏱️  Temps total: {inference_time:.2f}s")
        console.print(f"⚡ Temps moyen par fichier: {inference_time/len(audio_files):.3f}s")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Erreur inférence: {e}")
        return False

def test_streaming_inference(model):
    """Tester l'inférence en streaming (simulation)"""
    console.print("\n🌊 Test d'inférence streaming...", style="bold blue")
    
    try:
        # Créer un signal audio long
        sample_rate = 16000
        duration = 5.0
        num_samples = int(sample_rate * duration)
        
        # Signal avec variations de fréquence dans le temps
        t = np.linspace(0, duration, num_samples)
        signal = np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t * (1 + 0.1 * t))
        
        # Simuler le streaming en découpant en chunks
        chunk_duration = 0.5  # 500ms chunks
        chunk_samples = int(sample_rate * chunk_duration)
        
        console.print(f"🔄 Simulation streaming par chunks de {chunk_duration}s...")
        
        accumulated_audio = []
        
        with console.status("[bold green]Processing chunks...") as status:
            for i in range(0, num_samples, chunk_samples):
                chunk = signal[i:i + chunk_samples]
                accumulated_audio.extend(chunk)
                
                # Traiter l'audio accumulé tous les 2 chunks
                if (i // chunk_samples) % 2 == 1:
                    current_audio = np.array(accumulated_audio)
                    
                    # Test d'inférence
                    start_time = time.time()
                    transcription = model.transcribe([current_audio])
                    process_time = time.time() - start_time
                    
                    chunk_num = i // chunk_samples + 1
                    audio_duration = len(current_audio) / sample_rate
                    
                    status.update(f"Chunk {chunk_num}: {audio_duration:.1f}s audio -> {process_time:.3f}s processing")
                    
                    # Vérifier la latence
                    latency_ok = "✅" if process_time < chunk_duration else "⚠️"
                    console.print(f"{latency_ok} Chunk {chunk_num}: {process_time:.3f}s processing time")
        
        console.print("✅ Test streaming terminé")
        return True
        
    except Exception as e:
        console.print(f"❌ Erreur streaming: {e}")
        return False

def benchmark_model(model):
    """Benchmarker les performances du modèle"""
    console.print("\n📊 Benchmark des performances...", style="bold blue")
    
    try:
        durations = [1.0, 2.0, 5.0, 10.0]  # Différentes durées
        sample_rate = 16000
        
        results = []
        
        for duration in durations:
            num_samples = int(sample_rate * duration)
            
            # Créer signal test
            t = np.linspace(0, duration, num_samples)
            signal = np.sin(2 * np.pi * 300 * t) + 0.3 * np.random.normal(0, 1, num_samples)
            signal = signal / np.max(np.abs(signal))
            
            # Mesurer le temps d'inférence
            start_time = time.time()
            transcription = model.transcribe([signal])
            process_time = time.time() - start_time
            
            # Calculer le Real-Time Factor (RTF)
            rtf = process_time / duration
            
            results.append({
                'duration': duration,
                'process_time': process_time,
                'rtf': rtf,
                'transcription': transcription[0]
            })
        
        # Afficher les résultats
        table = Table(title="Benchmark Performance")
        table.add_column("Durée Audio", style="cyan")
        table.add_column("Temps Process", style="yellow")
        table.add_column("RTF", style="magenta")
        table.add_column("Performance", style="green")
        
        for result in results:
            duration = f"{result['duration']:.1f}s"
            process_time = f"{result['process_time']:.3f}s"
            rtf = f"{result['rtf']:.3f}"
            
            # Performance indicator
            if result['rtf'] < 0.1:
                perf = "🚀 Excellent"
            elif result['rtf'] < 0.3:
                perf = "✅ Bon"
            elif result['rtf'] < 1.0:
                perf = "⚠️ Acceptable"
            else:
                perf = "🔴 Lent"
            
            table.add_row(duration, process_time, rtf, perf)
        
        console.print(table)
        
        # Moyenne RTF
        avg_rtf = sum(r['rtf'] for r in results) / len(results)
        console.print(f"\n📈 RTF moyen: {avg_rtf:.3f}")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Erreur benchmark: {e}")
        return False

def test_french_vocabulary(model):
    """Tester la reconnaissance de vocabulaire français spécialisé"""
    console.print("\n🇫🇷 Test du vocabulaire français ferroviaire...", style="bold blue")
    
    # Mots clés ferroviaires français à tester
    french_vocab = [
        "correspondance", "terminus", "quai", "voie", 
        "Lyon", "Bordeaux", "Chalon", "Chauny",
        "TER", "TGV", "SNCF", "gare"
    ]
    
    try:
        # Récupérer le vocabulaire du modèle
        model_vocab = set()
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'vocabulary'):
            model_vocab = set(model.decoder.vocabulary)
        
        # Analyser la couverture
        table = Table(title="Analyse Vocabulaire Français")
        table.add_column("Mot", style="cyan")
        table.add_column("Dans Modèle", style="green")
        table.add_column("Variants Trouvés", style="yellow")
        
        coverage_count = 0
        
        for word in french_vocab:
            # Chercher le mot exact
            exact_match = word.lower() in [v.lower() for v in model_vocab]
            
            # Chercher des variants
            variants = [v for v in model_vocab if word.lower() in v.lower() or v.lower() in word.lower()]
            
            if exact_match:
                coverage_count += 1
                status = "✅ Oui"
            else:
                status = "❌ Non"
            
            variants_str = ", ".join(variants[:3]) if variants else "Aucun"
            if len(variants) > 3:
                variants_str += "..."
                
            table.add_row(word, status, variants_str)
        
        console.print(table)
        
        coverage_percent = (coverage_count / len(french_vocab)) * 100
        console.print(f"\n📊 Couverture vocabulaire: {coverage_count}/{len(french_vocab)} ({coverage_percent:.1f}%)")
        
        if coverage_percent < 50:
            console.print("⚠️  Faible couverture - Fine-tuning recommandé", style="bold yellow")
        else:
            console.print("✅ Couverture acceptable", style="bold green")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Erreur test vocabulaire: {e}")
        return False

def main():
    """Fonction principale"""
    console.print("🎯 Test d'inférence Conformer RNN-T", style="bold")
    console.print("=" * 50)
    
    # Charger le modèle
    model = load_model()
    if model is None:
        return False
    
    # Créer des fichiers audio de test
    audio_files = create_test_audio()
    
    # Tests d'inférence
    tests = [
        ("Inférence en lot", lambda: test_batch_inference(model, audio_files)),
        ("Streaming simulation", lambda: test_streaming_inference(model)),
        ("Benchmark performance", lambda: benchmark_model(model)),
        ("Vocabulaire français", lambda: test_french_vocabulary(model))
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        console.print(f"\n🧪 {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            console.print(f"❌ Erreur {test_name}: {e}")
            results[test_name] = False
    
    # Résumé final
    console.print("\n📋 Résumé des tests:", style="bold")
    
    for test_name, success in results.items():
        status = "✅" if success else "❌"
        console.print(f"  {status} {test_name}")
    
    success_count = sum(results.values())
    total_tests = len(results)
    
    console.print(f"\n🎯 Score: {success_count}/{total_tests} tests réussis")
    
    if success_count == total_tests:
        console.print("🎉 Tous les tests sont passés!", style="bold green")
        console.print("\nÉtapes suivantes:")
        console.print("1. Préparer vos données: python scripts/04_prepare_data.py")
        console.print("2. Lancer un fine-tuning test: python scripts/05_test_training.py")
    else:
        console.print("⚠️ Certains tests ont échoué. Vérifiez la configuration.", style="bold yellow")
    
    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)