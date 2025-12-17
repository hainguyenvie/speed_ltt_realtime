from pyannote.audio import Pipeline
import soundfile as sf
import torch
import numpy as np

p = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1')
a, sr = sf.read('recording.wav')

if len(a.shape) > 1:
    a = a.mean(axis=1)

w = torch.from_numpy(a[np.newaxis, :]).float()
d = p({'waveform': w, 'sample_rate': sr})

print('Type:', type(d))
print('Has itertracks:', hasattr(d, 'itertracks'))
print('Has segments:', hasattr(d, 'segments'))
print('\nPublic attributes:')
attrs = [x for x in dir(d) if not x.startswith('_')]
for attr in attrs[:30]:
    print(f'  - {attr}')

print('\nTrying to iterate:')
try:
    for segment, track, speaker in d.itertracks(yield_label=True):
        print(f'  {speaker}: {segment.start:.1f}s - {segment.end:.1f}s')
        break
except Exception as e:
    print(f'  itertracks error: {e}')

try:
    print('\nDirect iteration:')
    for item in d:
        print(f'  {item}')
        break
except Exception as e:
    print(f'  Direct iteration error: {e}')

print('\n Object:', d)
