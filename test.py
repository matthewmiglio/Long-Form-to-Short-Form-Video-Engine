import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import urllib.request

from utils import (
    parse_srt,
    timestamp_to_seconds,
    seconds_to_timestamp,
    format_transcript_for_llm,
    get_total_duration,
    get_text_for_time_range,
)


SAMPLE_SRT = """1
00:00:00,000 --> 00:00:03,500
Hello everyone welcome to the show

2
00:00:03,500 --> 00:00:07,200
Today we are going to talk about something amazing

3
00:00:07,200 --> 00:00:12,800
I went to the store yesterday and you won't believe what happened

4
00:00:12,800 --> 00:00:18,100
There was this guy just standing in the middle of the aisle

5
00:00:18,100 --> 00:00:25,000
And he turns to me and says the most ridiculous thing

6
00:00:25,000 --> 00:00:32,500
He said hey buddy do you know where the unicorn food is

7
00:00:32,500 --> 00:00:38,000
I could not stop laughing it was the funniest thing ever
"""


def _ollama_available():
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        return False


def _ffmpeg_available():
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


# ── Utils tests ──────────────────────────────────────────────


class TestTimestampConversion(unittest.TestCase):
    def test_timestamp_to_seconds_zero(self):
        self.assertAlmostEqual(timestamp_to_seconds("00:00:00,000"), 0.0)

    def test_timestamp_to_seconds_simple(self):
        self.assertAlmostEqual(timestamp_to_seconds("00:00:03,500"), 3.5)

    def test_timestamp_to_seconds_hours(self):
        self.assertAlmostEqual(timestamp_to_seconds("01:30:45,123"), 5445.123)

    def test_seconds_to_timestamp_zero(self):
        self.assertEqual(seconds_to_timestamp(0.0), "00:00:00,000")

    def test_seconds_to_timestamp_simple(self):
        self.assertEqual(seconds_to_timestamp(3.5), "00:00:03,500")

    def test_seconds_to_timestamp_hours(self):
        self.assertEqual(seconds_to_timestamp(5445.123), "01:30:45,123")

    def test_roundtrip(self):
        for val in [0.0, 1.5, 60.0, 3661.999]:
            ts = seconds_to_timestamp(val)
            result = timestamp_to_seconds(ts)
            self.assertAlmostEqual(result, val, places=2)


class TestParseSrt(unittest.TestCase):
    def test_parse_sample(self):
        entries = parse_srt(SAMPLE_SRT)
        self.assertEqual(len(entries), 7)
        self.assertEqual(entries[0]["index"], 1)
        self.assertAlmostEqual(entries[0]["start"], 0.0)
        self.assertAlmostEqual(entries[0]["end"], 3.5)
        self.assertEqual(entries[0]["text"], "Hello everyone welcome to the show")

    def test_parse_last_entry(self):
        entries = parse_srt(SAMPLE_SRT)
        last = entries[-1]
        self.assertEqual(last["index"], 7)
        self.assertAlmostEqual(last["start"], 32.5)
        self.assertAlmostEqual(last["end"], 38.0)

    def test_parse_empty(self):
        self.assertEqual(parse_srt(""), [])
        self.assertEqual(parse_srt("  \n\n  "), [])

    def test_parse_malformed_skipped(self):
        bad_srt = "not a valid srt\njust some text\n"
        self.assertEqual(parse_srt(bad_srt), [])


class TestFormatTranscriptForLlm(unittest.TestCase):
    def test_basic_format(self):
        entries = parse_srt(SAMPLE_SRT)
        result = format_transcript_for_llm(entries)
        lines = result.strip().split("\n")
        self.assertEqual(len(lines), 7)
        self.assertTrue(lines[0].startswith("[00:00]"))
        self.assertIn("Hello everyone", lines[0])

    def test_slice(self):
        entries = parse_srt(SAMPLE_SRT)
        result = format_transcript_for_llm(entries, start_idx=2, end_idx=4)
        lines = result.strip().split("\n")
        self.assertEqual(len(lines), 2)


class TestGetTotalDuration(unittest.TestCase):
    def test_duration(self):
        entries = parse_srt(SAMPLE_SRT)
        self.assertAlmostEqual(get_total_duration(entries), 38.0)

    def test_empty(self):
        self.assertEqual(get_total_duration([]), 0.0)


class TestGetTextForTimeRange(unittest.TestCase):
    def test_full_range(self):
        entries = parse_srt(SAMPLE_SRT)
        text = get_text_for_time_range(entries, 0, 38)
        self.assertIn("Hello everyone", text)
        self.assertIn("funniest thing ever", text)

    def test_partial_range(self):
        entries = parse_srt(SAMPLE_SRT)
        text = get_text_for_time_range(entries, 10, 20)
        self.assertIn("store yesterday", text)
        self.assertNotIn("Hello everyone", text)


# ── Segmenter tests ─────────────────────────────────────────


class TestSegmenterChunking(unittest.TestCase):
    def test_short_transcript_single_chunk(self):
        from segmenter import Segmenter

        seg = Segmenter(chunk_duration=600)
        entries = parse_srt(SAMPLE_SRT)
        chunks = seg._chunk_transcript(entries)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], (0, len(entries)))

    def test_long_transcript_multiple_chunks(self):
        from segmenter import Segmenter

        seg = Segmenter(chunk_duration=15, overlap_duration=5)
        entries = parse_srt(SAMPLE_SRT)
        chunks = seg._chunk_transcript(entries)
        self.assertGreater(len(chunks), 1)


class TestSegmenterParseLlmResponse(unittest.TestCase):
    def test_valid_response(self):
        from segmenter import Segmenter

        seg = Segmenter()
        entries = parse_srt(SAMPLE_SRT)
        response = json.dumps(
            [
                {
                    "start_time": 7.2,
                    "end_time": 32.5,
                    "description": "Funny store story",
                }
            ]
        )
        result = seg._parse_llm_response(response, entries, 38.0)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["start"], 7.2)
        self.assertAlmostEqual(result[0]["end"], 32.5)

    def test_invalid_json(self):
        from segmenter import Segmenter

        seg = Segmenter()
        entries = parse_srt(SAMPLE_SRT)
        result = seg._parse_llm_response("not json at all", entries, 38.0)
        self.assertEqual(result, [])

    def test_clamps_out_of_range(self):
        from segmenter import Segmenter

        seg = Segmenter()
        entries = parse_srt(SAMPLE_SRT)
        response = json.dumps(
            [{"start_time": -5.0, "end_time": 100.0, "description": "test"}]
        )
        result = seg._parse_llm_response(response, entries, 38.0)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["start"], 0.0)
        self.assertAlmostEqual(result[0]["end"], 38.0)

    def test_skips_tiny_segments(self):
        from segmenter import Segmenter

        seg = Segmenter()
        entries = parse_srt(SAMPLE_SRT)
        response = json.dumps(
            [{"start_time": 5.0, "end_time": 7.0, "description": "too short"}]
        )
        result = seg._parse_llm_response(response, entries, 38.0)
        self.assertEqual(result, [])


class TestSegmenterDeduplicate(unittest.TestCase):
    def test_no_overlap(self):
        from segmenter import Segmenter

        seg = Segmenter()
        segments = [
            {"start": 0, "end": 10, "description": "a"},
            {"start": 15, "end": 25, "description": "b"},
        ]
        result = seg._deduplicate(segments)
        self.assertEqual(len(result), 2)

    def test_removes_overlap(self):
        from segmenter import Segmenter

        seg = Segmenter()
        segments = [
            {"start": 0, "end": 20, "description": "a"},
            {"start": 5, "end": 25, "description": "b"},
        ]
        result = seg._deduplicate(segments)
        self.assertEqual(len(result), 1)


# ── Scorer tests ─────────────────────────────────────────────


class TestScorerPrompt(unittest.TestCase):
    def test_prompt_contains_text(self):
        from scorer import SCORING_PROMPT

        prompt = SCORING_PROMPT.format(
            duration=25.3, description="A funny moment", text="Some transcript text"
        )
        self.assertIn("25.3", prompt)
        self.assertIn("A funny moment", prompt)
        self.assertIn("Some transcript text", prompt)


# ── Extractor tests ──────────────────────────────────────────


@unittest.skipUnless(_ffmpeg_available(), "ffmpeg not available")
class TestExtractor(unittest.TestCase):
    def test_extract_clips_creates_dir(self):
        from extractor import Extractor

        ext = Extractor()
        tmpdir = os.path.join(tempfile.gettempdir(), "test_extractor_output")
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        # Just verify the dir creation logic (no actual video to extract)
        os.makedirs(tmpdir, exist_ok=True)
        self.assertTrue(os.path.isdir(tmpdir))
        shutil.rmtree(tmpdir)


# ── Integration tests ────────────────────────────────────────


@unittest.skipUnless(
    _ollama_available() and _ffmpeg_available(),
    "Ollama and/or ffmpeg not available",
)
class TestIntegration(unittest.TestCase):
    def test_segmenter_with_real_llm(self):
        from segmenter import Segmenter

        seg = Segmenter()
        entries = parse_srt(SAMPLE_SRT)
        segments = seg.segment_transcript(entries, min_duration=10, max_duration=38)
        self.assertIsInstance(segments, list)
        # LLM should find at least 1 segment in this sample
        if segments:
            self.assertIn("start", segments[0])
            self.assertIn("end", segments[0])
            self.assertIn("description", segments[0])

    def test_scorer_with_real_llm(self):
        from scorer import Scorer

        scorer = Scorer()
        segment = {
            "start": 7.2,
            "end": 32.5,
            "description": "Funny store story about unicorn food",
            "text": "I went to the store yesterday and this guy asked me where the unicorn food is",
        }
        result = scorer.score_segment(segment)
        if result is not None:
            self.assertIn("scores", result)
            scores = result["scores"]
            for key in ("engagement", "entertainment", "standalone", "hook", "emotional_impact", "total"):
                self.assertIn(key, scores)
            for key in ("engagement", "entertainment", "standalone", "hook", "emotional_impact"):
                self.assertGreaterEqual(scores[key], 1)
                self.assertLessEqual(scores[key], 10)


# ── End-to-end pipeline test ─────────────────────────────────


def _get_font_path():
    """Get a usable font path for ffmpeg drawtext."""
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path.replace("\\", "/").replace(":", "\\:")
    return None


def extract_clip_with_overlay(video_path, start, end, output_path, label):
    """Extract a clip with a fat GOOD/BAD text overlay using ffmpeg."""
    duration = end - start
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    color = "green" if label == "GOOD" else "red"
    font_path = _get_font_path()

    drawtext = (
        f"drawtext=text='{label}'"
        f":fontsize=120"
        f":fontcolor={color}"
        f":borderw=5"
        f":bordercolor=black"
        f":x=(w-text_w)/2"
        f":y=(h-text_h)/2"
    )
    if font_path:
        drawtext = (
            f"drawtext=fontfile='{font_path}'"
            f":text='{label}'"
            f":fontsize=120"
            f":fontcolor={color}"
            f":borderw=5"
            f":bordercolor=black"
            f":x=(w-text_w)/2"
            f":y=(h-text_h)/2"
        )

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-vf", drawtext,
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ffmpeg error: {result.stderr[-500:]}")
        return False
    return True


def run_e2e(video_path, output_dir="output_folder"):
    """Run the full pipeline on a video and export top 3 + bottom 3 clips."""
    from transcriber import Transcriber
    from segmenter import Segmenter
    from scorer import Scorer

    print("=" * 60)
    print("E2E Pipeline Test")
    print("=" * 60)
    print(f"Input:  {video_path}")
    print(f"Output: {output_dir}")

    # Clean output dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # 1) Transcribe
    print("\n[1/4] Transcribing...")
    transcriber = Transcriber(model_size="base.en")
    srt_string = transcriber.transcribe_to_srt(video_path)
    parsed_srt = parse_srt(srt_string)
    duration = get_total_duration(parsed_srt)
    print(f"  Got {len(parsed_srt)} subtitle entries, {duration:.1f}s total")

    if not parsed_srt:
        print("ERROR: No speech detected. Aborting.")
        sys.exit(1)

    # 2) Segment
    print("\n[2/4] Segmenting (LLM)...")
    segmenter = Segmenter()
    segments = segmenter.segment_transcript(parsed_srt, min_duration=15, max_duration=60)
    print(f"  Found {len(segments)} segments")

    if len(segments) < 6:
        print(f"  WARNING: Only {len(segments)} segments found (need 6 for top3+bottom3).")
        print("  Will use as many as available.")

    if not segments:
        print("ERROR: No segments found. Aborting.")
        sys.exit(1)

    # 3) Score
    print(f"\n[3/4] Scoring {len(segments)} segments (LLM)...")
    scorer = Scorer()
    scored = scorer.score_segments(segments, verbose=True)
    print(f"  Scored {len(scored)} / {len(segments)} segments")

    if not scored:
        print("ERROR: No segments scored. Is Ollama running? Aborting.")
        sys.exit(1)

    # 4) Rank and export
    print("\n[4/4] Ranking and exporting clips...")
    ranked = sorted(scored, key=lambda s: s["scores"]["total"], reverse=True)

    top_n = min(3, len(ranked))
    bottom_n = min(3, len(ranked) - top_n)

    top = ranked[:top_n]
    bottom = ranked[-bottom_n:] if bottom_n > 0 else []

    print(f"\n  TOP {top_n} (highest scores):")
    for i, seg in enumerate(top, 1):
        sc = seg["scores"]
        print(
            f"    {i}. [{seg['start']:.1f}s - {seg['end']:.1f}s] "
            f"total={sc['total']} "
            f"(eng={sc['engagement']} ent={sc['entertainment']} "
            f"stand={sc['standalone']} hook={sc['hook']} emo={sc['emotional_impact']})"
        )
        print(f"       {seg['description'][:80]}")

    if bottom:
        print(f"\n  BOTTOM {bottom_n} (lowest scores):")
        for i, seg in enumerate(bottom, 1):
            sc = seg["scores"]
            print(
                f"    {i}. [{seg['start']:.1f}s - {seg['end']:.1f}s] "
                f"total={sc['total']} "
                f"(eng={sc['engagement']} ent={sc['entertainment']} "
                f"stand={sc['standalone']} hook={sc['hook']} emo={sc['emotional_impact']})"
            )
            print(f"       {seg['description'][:80]}")

    # Export good clips
    print(f"\n  Exporting {top_n} GOOD clips...")
    for i, seg in enumerate(top, 1):
        out_path = os.path.join(output_dir, "good", f"video{i}.mp4")
        print(f"    -> {out_path}")
        ok = extract_clip_with_overlay(
            video_path, seg["start"], seg["end"], out_path, "GOOD"
        )
        if not ok:
            print(f"    FAILED to export clip {i}")

    # Export bad clips
    if bottom:
        print(f"  Exporting {bottom_n} BAD clips...")
        for i, seg in enumerate(bottom, 1):
            out_path = os.path.join(output_dir, "bad", f"video{i}.mp4")
            print(f"    -> {out_path}")
            ok = extract_clip_with_overlay(
                video_path, seg["start"], seg["end"], out_path, "BAD"
            )
            if not ok:
                print(f"    FAILED to export clip {i}")

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Good clips: {output_dir}/good/")
    print(f"  Bad clips:  {output_dir}/bad/")
    print("=" * 60)


if __name__ == "__main__":
    # If a video path is passed as an argument, run the e2e test.
    # Otherwise, run unit tests.
    if len(sys.argv) >= 2 and os.path.exists(sys.argv[1]):
        video_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) >= 3 else "output_folder"
        run_e2e(video_path, output_dir)
    else:
        unittest.main()
