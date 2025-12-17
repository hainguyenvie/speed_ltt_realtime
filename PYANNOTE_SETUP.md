# Pyannote.audio Speaker Diarization Setup

## Quick Start

```bash
# Run multi-speaker transcription
python transcript_with_speakers_pyannote.py meeting.wav hf_YOUR_TOKEN_HERE
```

## Prerequisites

### 1. Get HuggingFace Token

1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: `speaker-diarization`
4. Type: `Read`
5. Click "Generate"
6. Copy the token (starts with `hf_`)

### 2. Accept Model License

1. Go to: https://huggingface.co/pyannote/speaker-diarization-3.1
2. Click "Agree and access repository"

## Installation

✅ Already installed! (`pyannote.audio` + dependencies)

## Usage

```bash
python transcript_with_speakers_pyannote.py <audio_file> <hf_token>
```

**Example:**
```bash
python transcript_with_speakers_pyannote.py recording.wav hf_xxxxx
```

## Output Example

```
================================================================================
📝 TRANSCRIPT WITH SPEAKERS
================================================================================

[00:02] SPEAKER_00: Chào mọi người, hôm nay chúng ta sẽ họp về dự án mới

[00:08] SPEAKER_01: Vâng, tôi đã chuẩn bị các tài liệu

[00:14] SPEAKER_00: Rất tốt, chúng ta bắt đầu nhé

[00:18] SPEAKER_02: Tôi có một số câu hỏi về timeline

================================================================================

Summary:
  • Total segments transcribed: 4
  • Speakers detected: 3
    - SPEAKER_00: 2 segments
    - SPEAKER_01: 1 segment
    - SPEAKER_02: 1 segment
```

## Features

✅ **Automatic Speaker Detection** - Detects number of speakers automatically  
✅ **Accurate Speaker Segmentation** - Uses pyannote.audio state-of-the-art models  
✅ **Language Agnostic** - Works with Vietnamese and any language  
✅ **Integrated STT** - Transcribes with Zipformer  
✅ **Timestamp Labels** - Shows when each person spoke  

## How It Works

1. **Speaker Diarization** (pyannote.audio)
   - Segments audio by speaker
   - Returns: (start_time, end_time, speaker_id) for each segment

2. **Speech-to-Text** (Zipformer)
   - Transcribes each segment individually

3. **Combine Results**
   - Match speaker labels with transcripts
   - Display chronologically

## Tuning Parameters

### Number of Speakers

By default, pyannote auto-detects. To force a specific count:

```python
diarization = diarization_pipeline(
    audio_file,
    num_speakers=2  # Force 2 speakers
)
```

### Min/Max Speakers

```python
diarization = diarization_pipeline(
    audio_file,
    min_speakers=2,
    max_speakers=5
)
```

## Token Storage (Optional)

To avoid typing token every time:

```bash
# Set environment variable
export HF_TOKEN=hf_xxxxx

# Or use huggingface-cli
huggingface-cli login
```

Then use in script:
```python
import os
hf_token = os.environ.get('HF_TOKEN')
```

## Troubleshooting

### Error: "Model not found" or "Access denied"

**Solution:**
1. Check token is valid
2. Accept model license at: https://huggingface.co/pyannote/speaker-diarization-3.1

### Error: "torch not found"

**Solution:** Already installed with pyannote.audio

### Poor speaker separation

**Tips:**
- Use clean audio (minimal background noise)
- Ensure speakers have distinct voices
- Avoid overlapping speech
- Use high-quality recording (16kHz minimum)

## Performance

- **Diarization**: ~1-2x realtime (for 10min audio: ~10-20s)
- **STT**: 46-95x realtime (very fast)
- **Total**: ~2-3x realtime including both steps

## Files

| File | Purpose |
|------|---------|
| `transcript_with_speakers_pyannote.py` | Main script |
| `PYANNOTE_SETUP.md` | This guide |

## Next Steps

1. Get HF token
2. Accept model license
3. Run script with your audio file
4. Enjoy multi-speaker transcripts!

---

**Powered by pyannote.audio + Zipformer** 🎙️
