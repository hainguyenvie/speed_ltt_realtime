# Meeting Transcript - Hybrid Approach

## Quick Start

### Live Mode (During Meeting)
Real-time transcript WITHOUT speaker labels:

```bash
python meeting_transcript.py live
```

**What happens:**
- ✅ Real-time streaming transcript
- ✅ Zero delay
- ✅ Saves audio to `transcripts/meeting_TIMESTAMP.wav`
- ✅ Saves live transcript to `transcripts/transcript_live_TIMESTAMP.txt`

**Output example:**
```
[00:00] Chào mọi người, hôm nay chúng ta họp về dự án mới
[00:05] Vâng, tôi đã chuẩn bị các tài liệu
[00:10] Rất tốt, chúng ta bắt đầu nhé
```

### Review Mode (After Meeting)
Add accurate speaker labels:

```bash
python meeting_transcript.py review transcripts/meeting_20231217_083000.wav
```

**What happens:**
- 🔍 Process entire recording
- 🎯 Accurate speaker diarization
- 📝 Saves final transcript with speaker labels

**Output example:**
```
[00:00] SPEAKER_00: Chào mọi người, hôm nay chúng ta họp về dự án mới
[00:05] SPEAKER_01: Vâng, tôi đã chuẩn bị các tài liệu  
[00:10] SPEAKER_00: Rất tốt, chúng ta bắt đầu nhé
```

## Workflow

```
┌─────────────┐
│ Live Mode   │ ──┐
│ (Meeting)   │   │
└─────────────┘   │
                  ├── Audio + Live Transcript
┌─────────────┐   │
│ Review Mode │ ──┘
│ (Later)     │ ──> Final Transcript with Speakers
└─────────────┘
```

## Features

### Live Mode
- ✅ **Zero delay**: Transcript appears immediately
- ✅ **No speaker labels**: Fast processing
- ✅ **Auto-save**: Audio + transcript saved automatically
- ✅ **Non-blocking**: Recording never stops

### Review Mode  
- ✅ **Accurate speakers**: Process entire recording at once
- ✅ **Consistent labels**: SPEAKER_00, SPEAKER_01, etc.
- ✅ **Detailed transcript**: With timestamps and speakers
- ✅ **Summary stats**: Speaker count and segments

## Output Files

All files saved in `transcripts/` directory:

| File | Description |
|------|-------------|
| `meeting_TIMESTAMP.wav` | Full audio recording |
| `transcript_live_TIMESTAMP.txt` | Live transcript (no speakers) |
| `transcript_final_TIMESTAMP.txt` | Final transcript (with speakers) |

## Examples

### Complete Workflow

```bash
# 1. Start meeting
python meeting_transcript.py live

# During meeting: Speak naturally
# Transcript appears in real-time
# Press Ctrl+C when done

# 2. Review with speaker labels (later)
python meeting_transcript.py review transcripts/meeting_20231217_083000.wav
```

### Output Comparison

**Live Transcript** (`transcript_live_*.txt`):
```
# Live Transcript - 20231217_083000
# Started: 2023-12-17 08:30:00

[00:00] Chào mọi người
[00:05] Vâng, tôi đã chuẩn bị
[00:10] Rất tốt
```

**Final Transcript** (`transcript_final_*.txt`):
```
# Final Transcript with Speaker Labels
# Processed: 2023-12-17 09:00:00
# Audio: transcripts/meeting_20231217_083000.wav
# Speakers: 2

[00:00] SPEAKER_00: Chào mọi người
[00:05] SPEAKER_01: Vâng, tôi đã chuẩn bị
[00:10] SPEAKER_00: Rất tốt

📊 Summary:
   Total speakers: 2
   - SPEAKER_00: 2 segments
   - SPEAKER_01: 1 segment
```

## Why Hybrid Approach?

### The Problem
- ⏱️  **Real-time** diarization → Inconsistent speaker labels
- 🎯 **Accurate** diarization → Need entire audio (no streaming)

### The Solution
- 🎥 **Live Mode**: Get transcript ASAP (what matters during meeting)
- 🔍 **Review Mode**: Get accurate speakers (when you have time)

### Benefits
- ✅ Best of both worlds
- ✅ Meeting participants can follow in real-time
- ✅ Accurate speaker attribution for records
- ✅ No compromise on speed or accuracy

## Tips

1. **During meeting**: Focus on Live Mode for real-time collaboration
2. **After meeting**: Run Review Mode for official records
3. **Multiple speakers**: Review Mode becomes more valuable
4. **Short segments**: Live Mode may be sufficient

## Technical Details

- **STT Model**: Zipformer (46x realtime)
- **Diarization**: pyannote.audio 3.1
- **VAD**: RMS-based (1.5s silence threshold)
- **Audio**: 16kHz WAV, mono

---

**Powered by Zipformer + Pyannote.audio** 🎙️
