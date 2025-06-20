railway_asr_project/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── __init__.py
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── inference_config.yaml
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── conformer_rnnt.py
│   │   └── model_utils.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   └── augmentation.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── callbacks.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py
│   │   └── streaming.py
│   └── utils/
│       ├── __init__.py
│       ├── audio_utils.py
│       └── text_utils.py
├── setup/
│   ├── setup_env.py
│   ├── download_model.py
│   ├── test_inference.py
│   ├── prepare_data.py
│   ├── test_training.py
│   └── evaluate_model.py
├── data/
│   ├── raw/
│   │   ├── audio/
│   │   └── transcripts/
│   ├── processed/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── manifests/
│       ├── train_manifest.json
│       ├── val_manifest.json
│       └── test_manifest.json
├── models/
│   ├── pretrained/
│   ├── checkpoints/
│   └── final/
├── logs/
│   ├── tensorboard/
│   └── training/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_analysis.ipynb
│   └── 03_results_visualization.ipynb
└── tests_results/
    ├── __init__.py
    ├── test_models.py
    ├── test_data.py
    └── test_inference.py