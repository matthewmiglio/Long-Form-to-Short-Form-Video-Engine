import re


def parse_srt(srt_string):
    """Parse SRT string into list of dicts with index, start, end, text."""
    entries = []
    blocks = re.split(r"\n\n+", srt_string.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue
        timestamp_match = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
            lines[1].strip(),
        )
        if not timestamp_match:
            continue
        start = timestamp_to_seconds(timestamp_match.group(1))
        end = timestamp_to_seconds(timestamp_match.group(2))
        text = " ".join(line.strip() for line in lines[2:])
        entries.append({"index": index, "start": start, "end": end, "text": text})
    return entries


def timestamp_to_seconds(timestamp):
    """Convert SRT timestamp 'HH:MM:SS,mmm' to float seconds."""
    match = re.match(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", timestamp)
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {timestamp}")
    h, m, s, ms = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def seconds_to_timestamp(seconds):
    """Convert float seconds to SRT timestamp 'HH:MM:SS,mmm'."""
    total_ms = int(seconds * 1000)
    h = total_ms // 3600000
    m = (total_ms % 3600000) // 60000
    s = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def format_transcript_for_llm(parsed_srt, start_idx=0, end_idx=None):
    """Format parsed SRT entries as '[MM:SS] text' lines for LLM consumption."""
    if end_idx is None:
        end_idx = len(parsed_srt)
    lines = []
    for entry in parsed_srt[start_idx:end_idx]:
        total_s = int(entry["start"])
        m, s = divmod(total_s, 60)
        lines.append(f"[{m:02}:{s:02}] {entry['text']}")
    return "\n".join(lines)


def get_total_duration(parsed_srt):
    """Get total duration of transcript in seconds."""
    if not parsed_srt:
        return 0.0
    return parsed_srt[-1]["end"]


def get_text_for_time_range(parsed_srt, start_time, end_time):
    """Get concatenated text from all SRT entries within a time range."""
    texts = []
    for entry in parsed_srt:
        if entry["end"] <= start_time:
            continue
        if entry["start"] >= end_time:
            break
        texts.append(entry["text"])
    return " ".join(texts)
