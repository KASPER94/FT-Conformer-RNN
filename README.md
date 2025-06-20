# FT-Conformer-RNN
fine tune Conformer TT RealTime for real

PHASE 1 — ENVIRONNEMENT TECHNIQUE
1️⃣ Installer les prérequis
    Vérifier que GPU est à jour :
        Télécharge les derniers drivers NVIDIA.
        Installe CUDA Toolkit 12.x
        Installe cuDNN
        check: nvcc --version

2️⃣ Installer Anaconda ou uv
    conda create -n nemo_stt python=3.10
    conda activate nemo_stt
    ou
    uv init 
    rm main.py
    (uv venv)
    uv venv .venv --python cpython@3.10.17
    source venv/bin/activate ou .venv/Script/activate
    uv python install 3.10

3️⃣ Installer PyTorch compatible GPU
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ou
    uv add torch torchvision torchaudio
    git clone https://github.com/NVIDIA/NeMo.git
cd NeMo
    cd NeMo
    uv pip install -e .
    uv pip install 'nemo_toolkit[asr]'
    uv pip install nemo_toolkit['all']
    check torch and GPU : python set_up_check.py

    si Windows installation spécifique:
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        uv pip install pytorch-lightning hydra-core omegaconf
        uv pip install librosa soundfile transformers datasets
        uv pip install nltk spacy
        uv pip install torchmetrics scipy numpy
        uv pip install lightning[extra]
        uv pip install --no-deps nemo_toolkit
        uv pip install inflect jiwer sacremoses
        uv pip install editdistance g2p-en webdataset braceexpand
        uv pip install sentencepiece  # Alternative à youtokentome

    test nemo = python test_nemo.py

PHASE 2 — LE DATASET FRANÇAIS
1️⃣ Télécharger Common Voice FR
    https://commonvoice.mozilla.org/fr/datasets

2️⃣ Préparer le dataset au format NeMo
    manifest files JSON.
    Exemple de ligne JSON :
        {"audio_filepath": "/path/to/file.wav", "duration": 3.45, "text": "ton texte en minuscule sans ponctuation"}

 PHASE 3 — LE MODELE DE BASE
 dataset test:  wget https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav
 ou 
 Invoke-WebRequest -Uri "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav" -OutFile "2086-149220-0033.wav"

 information: 
    https://huggingface.co/nvidia/stt_fr_conformer_transducer_large
    https://huggingface.co/Ilyes/wav2vec2-large-xlsr-53-french
    https://arxiv.org/abs/2005.08100
    https://paperswithcode.com/sota/speech-recognition-on-common-voice-french
    Wav2Vec not opti stay on Conformer/RNN-T
    https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html
    https://arxiv.org/pdf/2504.06963
 fine-tuner un model:
    wget https://huggingface.co/nvidia/stt_fr_conformer_transducer_large/blob/main/stt_fr_conformer_transducer_large.nemo

 PHASE 4 — FINE-TUNING
training:
    python scripts/train_speech_recognition_transducer.py \
    --config-path=conf \
    --config-name=conformer_transducer_small \
    model.train_ds.manifest_filepath=/path/to/your/train_manifest.json \
    model.validation_ds.manifest_filepath=/path/to/your/val_manifest.json \
    model.tokenizer.dir=/path/to/your/tokenizer_dir \
    exp_manager.exp_dir=/path/to/your/output/dir \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    trainer.precision=16

PHASE 5 — DÉPLOIEMENT EN STREAMING TEMPS RÉEL