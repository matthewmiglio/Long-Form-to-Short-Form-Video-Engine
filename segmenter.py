import json
import urllib.request
from utils import format_transcript_for_llm, get_text_for_time_range


SEGMENTATION_PROMPT = """You are analyzing a video transcript to identify the best self-contained moments for short-form clips.

Find segments that are between {min_duration} and {max_duration} seconds long. Each segment must:
1. Contain a complete thought, story, joke, argument, or emotional moment
2. Start and end at natural speech boundaries (not mid-sentence)
3. Work as a standalone clip without needing prior context
4. Be interesting, dramatic, funny, or emotionally compelling

Transcript with timestamps:
{transcript}

Output a JSON array of segments. Each segment has:
- "start_time": start timestamp in seconds (float)
- "end_time": end timestamp in seconds (float)
- "description": one-sentence summary of the moment

Output ONLY valid JSON, nothing else."""


class Segmenter:
    def __init__(
        self,
        ollama_url="http://localhost:11434/api/generate",
        model="gemma2:9b",
        chunk_duration=600.0,
        overlap_duration=30.0,
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration

    def segment_transcript(self, parsed_srt, min_duration=20.0, max_duration=40.0):
        """
        Identify natural 20-40s moments in a transcript using an LLM.
        Returns list of dicts: [{start, end, description, text}, ...]
        """
        if not parsed_srt:
            return []

        total_duration = parsed_srt[-1]["end"]
        chunks = self._chunk_transcript(parsed_srt)

        all_segments = []
        for chunk_start_idx, chunk_end_idx in chunks:
            transcript_text = format_transcript_for_llm(
                parsed_srt, chunk_start_idx, chunk_end_idx
            )
            prompt = SEGMENTATION_PROMPT.format(
                min_duration=int(min_duration),
                max_duration=int(max_duration),
                transcript=transcript_text,
            )
            response = self._call_ollama(prompt)
            if response is None:
                continue
            segments = self._parse_llm_response(response, parsed_srt, total_duration)
            all_segments.extend(segments)

        # Deduplicate overlapping segments from chunk boundaries
        all_segments = self._deduplicate(all_segments)

        # Attach full text to each segment
        for seg in all_segments:
            seg["text"] = get_text_for_time_range(parsed_srt, seg["start"], seg["end"])

        return all_segments

    def _chunk_transcript(self, parsed_srt):
        """Split transcript into overlapping index ranges for LLM processing."""
        total_duration = parsed_srt[-1]["end"]
        if total_duration <= self.chunk_duration:
            return [(0, len(parsed_srt))]

        chunks = []
        chunk_start_time = 0.0
        while chunk_start_time < total_duration:
            chunk_end_time = chunk_start_time + self.chunk_duration

            start_idx = 0
            for i, entry in enumerate(parsed_srt):
                if entry["start"] >= chunk_start_time:
                    start_idx = i
                    break

            end_idx = len(parsed_srt)
            for i, entry in enumerate(parsed_srt):
                if entry["start"] >= chunk_end_time:
                    end_idx = i
                    break

            if start_idx < end_idx:
                chunks.append((start_idx, end_idx))

            chunk_start_time += self.chunk_duration - self.overlap_duration

        return chunks

    def _call_ollama(self, prompt):
        """Call Ollama API. Returns parsed response string or None."""
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"num_gpu": 0},
            }
        ).encode("utf-8")

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
            print(f"Segmentation LLM call failed: {e}")
            return None

    def _parse_llm_response(self, response, parsed_srt, total_duration):
        """Parse LLM JSON response and validate timestamps."""
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            print(f"Failed to parse segmentation JSON: {response[:200]}")
            return []

        # Handle both direct array and wrapped object
        if isinstance(data, dict):
            for key in ("segments", "results", "clips", "moments"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            else:
                return []

        if not isinstance(data, list):
            return []

        segments = []
        for item in data:
            try:
                start = float(item.get("start_time", 0))
                end = float(item.get("end_time", 0))
                description = str(item.get("description", ""))
            except (TypeError, ValueError):
                continue

            # Clamp to valid range
            start = max(0.0, min(start, total_duration))
            end = max(0.0, min(end, total_duration))
            if end <= start or (end - start) < 5.0:
                continue

            segments.append(
                {"start": start, "end": end, "description": description}
            )

        return segments

    def _deduplicate(self, segments):
        """Remove segments that overlap >50% with a higher-ranked earlier segment."""
        if len(segments) <= 1:
            return segments

        # Sort by start time
        segments.sort(key=lambda s: s["start"])
        kept = [segments[0]]

        for seg in segments[1:]:
            overlap = False
            for existing in kept:
                overlap_start = max(seg["start"], existing["start"])
                overlap_end = min(seg["end"], existing["end"])
                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    seg_duration = seg["end"] - seg["start"]
                    if overlap_duration / seg_duration > 0.5:
                        overlap = True
                        break
            if not overlap:
                kept.append(seg)

        return kept
