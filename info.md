1️⃣ Le modèle 
    le gros avantage de ce modèle par rapport à Whisper.

    Le Conformer encoder est ici adapté en causal attention (streaming possible).

    Le Transducer (RNN-T) est nativement streaming-compatible :

    il prédit un token à chaque frame audio, sans devoir attendre la séquence entière.

2️⃣ RNN-T intégrer Conformer
    AUDIO INPUT
        ↓
    Conformer Encoder (frames → embeddings)
        ↓
    Prediction Network (RNN small)
        ↓
    Joint Network (combine encoder + prediction)
        ↓
    Text output (char / subword units)

Donc le Conformer améliore la qualité acoustique,
et le Transducer (RNN-T) gère l’alignement texte/audio séquentiel.

3️⃣ Pourquoi cette combinaison Conformer + RNN-T est très puissante

| Fonction                  | Pourquoi c'est optimal                     |
| ------------------------- | ------------------------------------------ |
| Streaming                 | Possible avec causal attention             |
| Phonétique fine           | Conformer capture les détails locaux       |
| Prosodie longue           | Self-attention capture le contexte global  |
| Temps réel faible latence | Transducer permet du prédiction à la volée |

