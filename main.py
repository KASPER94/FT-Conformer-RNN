"""
test du model nvidia nemo
"""

import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="nvidia/stt_en_fastconformer_ctc_large")

output = asr_model.transcribe(['./samples/2086-149220-0033.wav'])
print(output[0].text)
