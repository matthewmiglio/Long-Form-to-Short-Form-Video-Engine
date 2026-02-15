import os
import subprocess


class Extractor:
    def __init__(self, quality_preset="fast"):
        """
        :param quality_preset: ffmpeg preset (ultrafast, fast, medium, slow)
        """
        self.quality_preset = quality_preset

    def extract_clip(self, video_path, start_time, end_time, output_path):
        """
        Extract a single clip from video using ffmpeg.
        Returns True on success, False on failure.
        """
        duration = end_time - start_time
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_time),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", self.quality_preset,
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr[-300:]}")
            return False
        return True

    def extract_clips(self, video_path, segments, output_dir, verbose=False):
        """
        Extract multiple clips from ranked segments.
        Returns list of output file paths for successful extractions.
        """
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []

        for i, segment in enumerate(segments):
            score = segment.get("scores", {}).get("total", 0)
            filename = f"clip_{i + 1:03d}_score_{score}.mp4"
            output_path = os.path.join(output_dir, filename)

            if verbose:
                print(
                    f"  Extracting clip {i + 1}/{len(segments)}: "
                    f"[{segment['start']:.1f}s - {segment['end']:.1f}s] "
                    f"score={score}"
                )

            if self.extract_clip(
                video_path, segment["start"], segment["end"], output_path
            ):
                output_paths.append(output_path)
            else:
                print(f"  Failed to extract clip {i + 1}")

        return output_paths
