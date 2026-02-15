import json
import urllib.request


SCORING_PROMPT = """Score this video segment for short-form clip potential.
Output a JSON object with exactly these five integer fields (1-10 each):

- "engagement": How interesting, dramatic, or attention-grabbing is this?
  (1=boring/mundane, 10=extremely dramatic/compelling)
- "entertainment": How entertaining, funny, or enjoyable is this?
  (1=dry/dull, 10=hilarious/captivating)
- "standalone": How well does this work as a standalone clip without prior context?
  (1=completely requires context, 10=fully self-contained and clear)
- "hook": How quickly does it grab attention in the first few seconds?
  (1=very slow buildup, 10=instant hook)
- "emotional_impact": How much emotional resonance does it carry?
  (1=emotionally flat, 10=powerful emotional punch)

Segment duration: {duration:.1f}s
Segment description: {description}
Segment transcript:
{text}

Output ONLY valid JSON, nothing else."""


class Scorer:
    def __init__(
        self,
        ollama_url="http://localhost:11434/api/generate",
        model="gemma2:9b",
    ):
        self.ollama_url = ollama_url
        self.model = model

    def score_segment(self, segment):
        """
        Score a single segment. Returns segment dict with 'scores' added,
        or None on failure.
        """
        duration = segment["end"] - segment["start"]
        prompt = SCORING_PROMPT.format(
            duration=duration,
            description=segment.get("description", ""),
            text=segment.get("text", "")[:1500],
        )

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
                raw_scores = json.loads(result["response"])

            scores = {
                "engagement": max(1, min(10, int(raw_scores.get("engagement", 5)))),
                "entertainment": max(1, min(10, int(raw_scores.get("entertainment", 5)))),
                "standalone": max(1, min(10, int(raw_scores.get("standalone", 5)))),
                "hook": max(1, min(10, int(raw_scores.get("hook", 5)))),
                "emotional_impact": max(1, min(10, int(raw_scores.get("emotional_impact", 5)))),
            }
            scores["total"] = sum(scores.values())

            return {**segment, "scores": scores}

        except Exception as e:
            print(f"Scoring failed: {e}")
            return None

    def score_segments(self, segments, verbose=False):
        """Score multiple segments. Returns list of successfully scored segments."""
        scored = []
        total = len(segments)
        for i, segment in enumerate(segments):
            if verbose:
                print(f"  Scoring segment {i + 1}/{total}...")
            result = self.score_segment(segment)
            if result is not None:
                scored.append(result)
        return scored
