import json
import re
import subprocess
import urllib.request
from utils import get_text_for_time_range


FILTER_PROMPT = """You are selecting the best short-form video clips from a longer video.

Below are {count} candidate segments. Each segment starts and ends at a natural
boundary (audio pause or scene change), so clips won't cut off mid-sentence.

{candidates_text}

Select the {top_k} most compelling segments for short-form clips. Prioritize:
1. Complete, self-contained stories, jokes, arguments, or emotional moments
2. Strong opening hooks that grab attention immediately
3. Drama, humor, conflict, or emotional resonance
4. Works WITHOUT any prior context

Return a JSON object with a "selected" field containing an array of segment numbers.
Example: {{"selected": [3, 7, 12]}}
Output ONLY valid JSON."""


class Segmenter:
    def __init__(
        self,
        ollama_url="http://localhost:11434/api/generate",
        model="gemma2:9b",
        scene_threshold=0.3,
        silence_noise_db=-30,
        silence_min_duration=0.5,
        boundary_merge_radius=2.0,
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.scene_threshold = scene_threshold
        self.silence_noise_db = silence_noise_db
        self.silence_min_duration = silence_min_duration
        self.boundary_merge_radius = boundary_merge_radius

    def segment_transcript(
        self, video_path, parsed_srt, min_duration=15.0, max_duration=60.0
    ):
        """
        Two-pass segmentation:
          Pass 1 — ffmpeg scene + silence detection → natural boundary points
          Pass 2 — generate candidate segments from boundary pairs
          Pass 3 — LLM filters to the best candidates
        """
        if not parsed_srt:
            return []

        total_duration = parsed_srt[-1]["end"]

        # Pass 1: detect natural boundaries
        print("    Detecting scene changes...")
        scene_times = self._detect_scenes(video_path)
        print(f"    Found {len(scene_times)} scene changes")

        print("    Detecting silence gaps...")
        silence_times = self._detect_silences(video_path)
        print(f"    Found {len(silence_times)} silence points")

        boundaries = self._merge_boundaries(
            scene_times, silence_times, total_duration
        )
        print(f"    {len(boundaries)} boundary points after merging")

        # Pass 2: build candidate segments
        candidates = self._build_candidates(
            boundaries, parsed_srt, min_duration, max_duration
        )
        print(f"    Generated {len(candidates)} candidate segments")

        if not candidates:
            return []

        # Pass 3: LLM selects best candidates
        print("    LLM filtering candidates...")
        selected = self._llm_filter(candidates, max_results=20)
        print(f"    Selected {len(selected)} segments")

        selected = self._deduplicate(selected)
        return selected

    # ── ffmpeg detection ─────────────────────────────────────

    def _detect_scenes(self, video_path):
        """Use ffmpeg scene-change detection (visual cuts)."""
        vf = f"fps=2,select=gt(scene\\,{self.scene_threshold}),showinfo"
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", vf,
            "-vsync", "vfr", "-an",
            "-f", "null", "-",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

        times = []
        for line in result.stderr.split("\n"):
            if "showinfo" not in line.lower():
                continue
            m = re.search(r"pts_time:\s*([0-9.]+)", line)
            if m:
                times.append(float(m.group(1)))
        return times

    def _detect_silences(self, video_path):
        """Use ffmpeg silence detection to find natural speech pauses."""
        af = (
            f"silencedetect=noise={self.silence_noise_db}dB"
            f":d={self.silence_min_duration}"
        )
        cmd = [
            "ffmpeg", "-i", video_path,
            "-af", af,
            "-vn", "-f", "null", "-",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

        times = []
        for line in result.stderr.split("\n"):
            # silence_end = where speech resumes (good clip-start point)
            m = re.search(r"silence_end:\s*([0-9.]+)", line)
            if m:
                times.append(float(m.group(1)))
            # silence_start = where speech stops (good clip-end point)
            m = re.search(r"silence_start:\s*([0-9.]+)", line)
            if m:
                times.append(float(m.group(1)))
        return times

    # ── boundary merging ─────────────────────────────────────

    def _merge_boundaries(self, scene_times, silence_times, total_duration):
        """Merge all boundary points, deduplicate within merge_radius."""
        all_times = sorted(
            set([0.0] + scene_times + silence_times + [total_duration])
        )
        if len(all_times) <= 2:
            return all_times

        merged = [all_times[0]]
        for t in all_times[1:]:
            if t - merged[-1] >= self.boundary_merge_radius:
                merged.append(t)

        # Ensure video end is included
        if merged[-1] < total_duration - 1.0:
            merged.append(total_duration)

        return merged

    # ── candidate generation ─────────────────────────────────

    def _build_candidates(
        self, boundaries, parsed_srt, min_dur, max_dur, max_candidates=100
    ):
        """Generate candidate segments from boundary pairs, pre-ranked by heuristic."""
        candidates = []
        target_dur = (min_dur + max_dur) / 2

        for i, start in enumerate(boundaries):
            for j in range(i + 1, len(boundaries)):
                end = boundaries[j]
                dur = end - start
                if dur < min_dur:
                    continue
                if dur > max_dur:
                    break

                text = get_text_for_time_range(parsed_srt, start, end)
                if not text.strip():
                    continue

                # Heuristic pre-score: prefer speech-dense segments near target duration
                word_count = len(text.split())
                word_density = word_count / dur if dur > 0 else 0
                duration_score = 1.0 - abs(dur - target_dur) / target_dur
                pre_score = word_density * 0.7 + duration_score * 0.3

                candidates.append({
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "text": text,
                    "description": "",
                    "_score": pre_score,
                })

        # Keep top candidates by heuristic
        if len(candidates) > max_candidates:
            candidates.sort(key=lambda c: c["_score"], reverse=True)
            candidates = candidates[:max_candidates]

        for c in candidates:
            del c["_score"]

        return candidates

    # ── LLM filtering ────────────────────────────────────────

    def _llm_filter(self, candidates, max_results=20, batch_size=25):
        """Send candidate batches to LLM, keep the best ones."""
        if len(candidates) <= max_results:
            for c in candidates:
                if not c["description"]:
                    c["description"] = c["text"][:100]
            return candidates

        selected = []
        picks_per_batch = max(
            2, (max_results * batch_size) // len(candidates) + 1
        )

        for batch_start in range(0, len(candidates), batch_size):
            batch = candidates[batch_start : batch_start + batch_size]
            picked_indices = self._llm_pick_best(batch, picks_per_batch)
            for idx in picked_indices:
                if 0 <= idx < len(batch):
                    seg = batch[idx]
                    if not seg["description"]:
                        seg["description"] = seg["text"][:100]
                    selected.append(seg)

        return selected[:max_results]

    def _llm_pick_best(self, candidates, top_k):
        """Ask LLM to pick the top_k most interesting candidates from a batch."""
        lines = []
        for i, c in enumerate(candidates, 1):
            dur = c["end"] - c["start"]
            m_s, s_s = divmod(int(c["start"]), 60)
            m_e, s_e = divmod(int(c["end"]), 60)
            snippet = c["text"][:300]
            lines.append(
                f"Segment {i}: [{m_s}:{s_s:02d} - {m_e}:{s_e:02d}] ({dur:.0f}s)\n"
                f"  {snippet}"
            )

        prompt = FILTER_PROMPT.format(
            count=len(candidates),
            top_k=top_k,
            candidates_text="\n\n".join(lines),
        )

        response = self._call_ollama(prompt)
        if response is None:
            return list(range(min(top_k, len(candidates))))

        try:
            data = json.loads(response)
            if isinstance(data, dict):
                for key in ("selected", "segments", "results", "indices"):
                    if key in data and isinstance(data[key], list):
                        data = data[key]
                        break
                else:
                    return list(range(min(top_k, len(candidates))))
            if isinstance(data, list):
                return [
                    int(x) - 1
                    for x in data
                    if isinstance(x, (int, float))
                ]
        except (json.JSONDecodeError, ValueError):
            pass

        return list(range(min(top_k, len(candidates))))

    # ── Ollama call ──────────────────────────────────────────

    def _call_ollama(self, prompt):
        """Call Ollama API and return the response string."""
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"num_gpu": 0},
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                self.ollama_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["response"]
        except Exception as e:
            print(f"    LLM call failed: {e}")
            return None

    # ── deduplication ────────────────────────────────────────

    def _deduplicate(self, segments):
        """Remove segments that overlap >50% with a higher-ranked earlier one."""
        if len(segments) <= 1:
            return segments

        segments.sort(key=lambda s: s["start"])
        kept = [segments[0]]

        for seg in segments[1:]:
            dominated = False
            for existing in kept:
                overlap_start = max(seg["start"], existing["start"])
                overlap_end = min(seg["end"], existing["end"])
                if overlap_end > overlap_start:
                    overlap_dur = overlap_end - overlap_start
                    seg_dur = seg["end"] - seg["start"]
                    if overlap_dur / seg_dur > 0.5:
                        dominated = True
                        break
            if not dominated:
                kept.append(seg)

        return kept
