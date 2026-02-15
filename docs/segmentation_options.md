# Segmentation Strategy Options

## Problem

The original LLM-only segmentation (Option A below) hallucinates timestamps from text, causing clips to cut off mid-sentence or mid-thought.

## Options Considered

| # | Strategy | How it works | Pros | Cons |
|---|----------|-------------|------|------|
| **A** | **LLM-only (original)** | Send transcript chunks to LLM, ask it to return start/end timestamps for interesting moments. | Simple, single-pass. | LLM guesses timestamps from text — frequently cuts mid-sentence. |
| **B** | **Silence/pause-based splitting** | Use ffmpeg `silencedetect` to find natural pauses. LLM picks which pause-bounded segments are interesting. | Cuts always land on natural pauses. | Misses moments if pauses are rare. No visual awareness. |
| **C** | **Sentence-boundary segmentation** | Use Whisper word-level timestamps to build sentence boundaries. Generate all valid windows, score them. | Precise, leverages existing data. | Combinatorial explosion on long videos. |
| **D** | **Hybrid: LLM picks moments, snap to boundaries** | Keep LLM segmentation, but snap start/end to nearest sentence boundary via Whisper word timestamps. | Minimal code change. | Still depends on LLM timestamp accuracy for general region. |
| **E** | **Two-pass: scene detection + silence detection + LLM filter** | Pass 1: ffmpeg `scenedetect` + `silencedetect` to find natural visual/audio boundaries. Pass 2: generate candidate segments from boundary pairs (15-60s). Pass 3: LLM filters/ranks candidates. | Visual + audio awareness. Clips always start/end at natural breaks. Best quality. | Heavier first pass (ffmpeg decode). |

## Selected: Option E

### Pipeline

1. **Scene detection** — `ffmpeg -vf "fps=2,select=gt(scene,T),showinfo"` finds visual scene changes
2. **Silence detection** — `ffmpeg -af "silencedetect=noise=-30dB:d=0.5"` finds audio pauses
3. **Merge boundaries** — combine scene + silence timestamps, deduplicate within 2s radius
4. **Build candidates** — all boundary pairs where duration is in [min_dur, max_dur], pre-ranked by word density + duration fit
5. **LLM filter** — send candidate batches to LLM, pick most compelling segments
6. **Deduplicate** — remove >50% overlapping segments
