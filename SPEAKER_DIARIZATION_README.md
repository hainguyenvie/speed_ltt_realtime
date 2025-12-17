# 🎭 Multi-Speaker Transcription

Automatically identify different speakers and transcribe what each person said using speaker diarization + speech-to-text.

## Quick Start

```bash
# 1. Download speaker diarization models
python download_diarization_models.py

# 2. Run multi-speaker transcription
python transcript_with_speakers.py meeting.wav
```

## Output Example

```
================================================================================
📝 TRANSCRIPT WITH SPEAKERS
================================================================================

[00:02] Speaker 1: Chào mọi người, hôm nay chúng ta sẽ họp về dự án mới

[00:08] Speaker 2: Vâng, tôi đã chuẩn bị các tài liệu

[00:14] Speaker 1: Rất tốt, chúng ta bắt đầu nhé

[00:18] Speaker 3: Tôi có một số câu hỏi về timeline

[00:25] Speaker 2: Để tôi giải thích chi tiết

================================================================================

Summary:
  • Total segments: 5
  • Speakers detected: 3
    - Speaker 1: 2 segments
    - Speaker 2: 2 segments
    - Speaker 3: 1 segment
```

## Features

✅ **Automatic Speaker Detection** - No need to specify number of speakers  
✅ **Speaker Segmentation** - Detects when speakers change  
✅ **Speaker Clustering** - Groups segments by same speaker  
✅ **Integrated STT** - Transcribes each segment  
✅ **Timestamp Labels** - Shows when each person spoke  

## How It Works

```
Audio File
    ↓
[1] Speaker Segmentation (pyannote)
    → Detect speech regions
    → Find speaker changes
    ↓
[2] Speaker Embedding (3D-Speaker)
    → Extract voice fingerprint
    → Cluster similar voices
    ↓
[3] Speech-to-Text (Zipformer)
    → Transcribe each segment
    ↓
[4] Combine Results
    → Match speaker ID + transcript
    ↓
Output: Speaker-labeled transcript
```

## Installation

### 1. Download Models

```bash
python download_diarization_models.py
```

This downloads (~75MB total):
- `pyannote-segmentation-3-0` (66MB) - Speaker segmentation
- `3D-Speaker embedding` (9.6MB) - Voice fingerprints

### 2. Dependencies

All dependencies should already be installed from Zipformer setup:
- `sherpa-onnx` ✅
- `numpy` ✅
- `wave` ✅ (built-in)

## Usage

### Basic Usage

```bash
python transcript_with_speakers.py audio.wav
```

### Python API

```python
from transcript_with_speakers import SpeakerDiarizer, ZipformerSTT

# Initialize
diarizer = SpeakerDiarizer()
stt = ZipformerSTT()

# Get speaker segments
segments = diarizer.process("meeting.wav")
# Returns: [(start, end, "Speaker 1"), (start, end, "Speaker 2"), ...]

# Transcribe each segment
for start, end, speaker in segments:
    segment_audio = extract_segment(audio, start, end)
    text = stt.transcribe_segment(segment_audio, sample_rate)
    print(f"{speaker}: {text}")
```

## Configuration

### Auto Speaker Detection

By default, the number of speakers is detected automatically:

```python
clustering=sherpa_onnx.FastClusteringConfig(
    num_clusters=-1,  # Auto-detect
    threshold=0.5,    # Similarity threshold (0.0-1.0)
)
```

### Fixed Number of Speakers

If you know there are exactly N speakers:

```python
clustering=sherpa_onnx.FastClusteringConfig(
    num_clusters=2,   # Force 2 speakers
    threshold=0.5,
)
```

### Tuning Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `threshold` | Speaker similarity threshold | 0.5 | 0.0-1.0 |
| `min_duration_on` | Min speech duration | 0.3s | 0.1-1.0s |
| `min_duration_off` | Min silence duration | 0.5s | 0.1-2.0s |

**Lower threshold** → More speakers (more sensitive)  
**Higher threshold** → Fewer speakers (less sensitive)

## Technical Details

### Speaker Segmentation

**Model**: pyannote/segmentation-3.0  
**Purpose**: Split audio into speech segments  
**Output**: Start/end times of each speech region  

### Speaker Embedding

**Model**: 3D-Speaker (alibaba-damo-academy)  
**Purpose**: Extract voice fingerprints  
**Output**: 192-dim embedding vector per segment  

### Speaker Clustering

**Algorithm**: Agglomerative Clustering  
**Purpose**: Group segments by same speaker  
**Output**: Speaker ID for each segment  

### Speech-to-Text

**Model**: Zipformer-30M-RNNT-6000h  
**Purpose**: Transcribe Vietnamese speech  
**Speed**: 46-95x realtime  

## Use Cases

| Use Case | Description |
|----------|-------------|
| **Meeting Transcription** | Who said what in meetings |
| **Interview Processing** | Separate interviewer/interviewee |
| **Podcast Transcripts** | Label host and guests |
| **Call Center** | Agent vs customer |
| **Multi-person Conversations** | Any multi-speaker scenario |

## Performance

- **Segmentation**: ~1-2x realtime
- **Embedding**: ~10x realtime
- **Clustering**: <1s for 10min audio
- **STT**: 46-95x realtime

**Total**: ~5-10x realtime (including all steps)

## Troubleshooting

### Wrong Number of Speakers

**Too many speakers detected:**
- Increase `threshold` (e.g., 0.6 or 0.7)
- Increase `min_duration_on` to filter short segments

**Too few speakers detected:**
- Decrease `threshold` (e.g., 0.3 or 0.4)
- Decrease `min_duration_off` for faster speaker changes

### Poor Speaker Separation

- Clean audio works best
- Reduce background noise
- Ensure speakers have distinct voices
- Avoid overlapping speech

### Model Not Found

```bash
python download_diarization_models.py
```

## Limitations

- ❌ **No overlapping speech** - Can't handle simultaneous speakers
- ❌ **No speaker names** - Only assigns Speaker 1, 2, 3... labels
- ⚠️ **Needs distinct voices** - Similar voices may be grouped together
- ⚠️ **Clean audio preferred** - Background noise affects accuracy

## Future Enhancements

- [ ] Speaker name assignment (with voice samples)
- [ ] Real-time diarization
- [ ] Overlapping speech handling
- [ ] Speaker enrollment (identify known speakers)

## Files

| File | Purpose |
|------|---------|
| `download_diarization_models.py` | Download speaker models |
| `transcript_with_speakers.py` | Main diarization + STT script |
| `speaker-diarization-models/` | Downloaded models directory |

## Credits

- **Segmentation**: [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- **Embedding**: [3D-Speaker](https://github.com/alibaba-damo-academy/3D-Speaker)
- **STT**: [Zipformer](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h)
- **Infrastructure**: [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)

---

**Made for Vietnamese Multi-Speaker Transcription** 🎙️
