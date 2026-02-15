import argparse
import os
import sys

from transcriber import Transcriber
from segmenter import Segmenter
from scorer import Scorer
from extractor import Extractor
from utils import parse_srt, get_total_duration


def run_pipeline(args):
    """Run the full clip extraction pipeline."""
    print("=" * 60)
    print("Long-Form to Short-Form Video Clip Extractor")
    print("=" * 60)

    # Step 1: Transcribe
    print("\n[1/5] Transcribing video...")
    transcriber = Transcriber(model_size=args.model)
    srt_string = transcriber.transcribe_to_srt(args.video_path, save=args.save_transcript)
    parsed_srt = parse_srt(srt_string)
    duration = get_total_duration(parsed_srt)
    print(f"  Transcribed {duration:.1f}s of audio ({len(parsed_srt)} segments)")

    if not parsed_srt:
        print("No speech detected in video.")
        return

    # Step 2: Segment
    print("\n[2/5] Identifying natural moments...")
    segmenter = Segmenter()
    segments = segmenter.segment_transcript(
        parsed_srt,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )
    print(f"  Found {len(segments)} potential clips")

    if not segments:
        print("No segments found. Try adjusting --min-duration / --max-duration.")
        return

    # Step 3: Score
    print(f"\n[3/5] Scoring {len(segments)} segments...")
    scorer = Scorer()
    scored_segments = scorer.score_segments(segments, verbose=args.verbose)
    print(f"  Successfully scored {len(scored_segments)} segments")

    if not scored_segments:
        print("No segments could be scored. Is Ollama running?")
        return

    # Step 4: Rank
    print("\n[4/5] Ranking segments...")
    ranked = sorted(scored_segments, key=lambda s: s["scores"]["total"], reverse=True)
    top = ranked[: args.top_n]
    bottom = ranked[-args.bottom_n :] if args.bottom_n else []

    print(f"\n  Top {len(top)}:")
    for i, seg in enumerate(top, 1):
        scores = seg["scores"]
        print(
            f"    {i}. [{seg['start']:.1f}s - {seg['end']:.1f}s] "
            f"score={scores['total']} | {seg['description'][:60]}"
        )

    if bottom:
        print(f"\n  Bottom {len(bottom)}:")
        for i, seg in enumerate(bottom, 1):
            scores = seg["scores"]
            print(
                f"    {i}. [{seg['start']:.1f}s - {seg['end']:.1f}s] "
                f"score={scores['total']} | {seg['description'][:60]}"
            )

    # Step 5: Extract
    extractor = Extractor()
    extracted = []

    top_dir = os.path.join(args.output_dir, "top")
    print(f"\n[5/5] Extracting top {len(top)} clips to {top_dir}...")
    extracted.extend(
        extractor.extract_clips(args.video_path, top, top_dir, verbose=args.verbose)
    )

    if bottom:
        bottom_dir = os.path.join(args.output_dir, "bottom")
        print(f"       Extracting bottom {len(bottom)} clips to {bottom_dir}...")
        extracted.extend(
            extractor.extract_clips(
                args.video_path, bottom, bottom_dir, verbose=args.verbose
            )
        )

    print(f"\nDone! Extracted {len(extracted)} clips to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract the best short-form clips from a long-form video.",
    )
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("top_n", type=int, help="Number of top clips to extract")
    parser.add_argument(
        "--bottom-n",
        type=int,
        default=0,
        help="Also extract the N worst-scoring clips (default: 0)",
    )
    parser.add_argument("output_dir", help="Directory to save extracted clips")
    parser.add_argument(
        "--model",
        default="base.en",
        help="Whisper model size (default: base.en)",
    )
    parser.add_argument("--verbose", action="store_true", help="Detailed progress output")
    parser.add_argument(
        "--save-transcript", action="store_true", help="Save SRT transcript file"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=20.0,
        help="Minimum clip duration in seconds (default: 20)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=40.0,
        help="Maximum clip duration in seconds (default: 40)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    if args.top_n < 1:
        print("Error: top_n must be at least 1")
        sys.exit(1)

    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
