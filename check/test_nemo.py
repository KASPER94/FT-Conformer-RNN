import nemo
import nemo.collections.asr as nemo_asr
import nemo.collections.nlp as nemo_nlp

print("✓ NeMo importé avec succès")
print("✓ Version NeMo:", nemo.__version__)
print("✓ Module ASR disponible")
print("✓ Module NLP disponible")

# Tester l'accès aux modèles pré-entraînés
try:
    # Lister quelques modèles disponibles
    asr_models = nemo_asr.models.ASRModel.list_available_models()
    print(f"✓ {len(asr_models)} modèles ASR disponibles")
    print("Exemples:", asr_models[:3])
except Exception as e:
    print("⚠ Problème avec les modèles ASR:", e)

print("\n🎉 NeMo est prêt à être utilisé !")
