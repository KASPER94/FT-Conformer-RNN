import torch
import torchaudio
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
import re

model_name = "Ilyes/wav2vec2-large-xlsr-53-french"

device = "cpu" # "cuda"

model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
processor = Wav2Vec2Processor.from_pretrained(model_name)

ds = load_dataset("common_voice", "fr", split="test", cache_dir="./data/fr")

chars_to_ignore_regex =  r'[\,\?\.\!\;\:\"\“\%\‘\”\�\‘\’\’\’\‘\…\·\!\ǃ\?\«\‹\»\›“\”\\ʿ\ʾ\„\∞\\|\.\,\;\:\*\—\–\─\―\_\/\:\ː\;\,\=\«\»\→]'
def map_to_array(batch):
    speech, _ = torchaudio.load(batch["path"])
    batch["speech"] = resampler.forward(speech.squeeze(0)).numpy()
    batch["sampling_rate"] = resampler.new_freq
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower().replace("’", "'")
    return batch
resampler = torchaudio.transforms.Resample(48_000, 16_000)
    
ds = ds.map(map_to_array)

def map_to_pred(batch):
    features = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0], padding=True, return_tensors="pt")
    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
    pred_ids = torch.argmax(logits, dim=-1)
    batch["predicted"] = processor.batch_decode(pred_ids)
    batch["target"] = batch["sentence"]
    print(batch["target"], " --> ", batch[sentence])
    return batch
    
result = ds.map(map_to_pred, batched=True, batch_size=16, remove_columns=list(ds.features.keys()))
wer = load_metric("wer")
print(wer.compute(predictions=result["predicted"], references=result["target"]))
