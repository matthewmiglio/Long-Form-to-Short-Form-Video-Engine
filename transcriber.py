import os
from pathlib import Path
from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self, model_size="base.en", compute_type="int8"):
        """
        Initialize the Faster-Whisper model.
        :param model_size: tiny.en, base.en, small.en, medium.en, large-v2
        :param compute_type: int8, float16, or float32
        """
        self.model = WhisperModel(model_size, compute_type=compute_type, device="cpu")

    def transcribe_to_srt(self, video_path, save=False, output_folder="transcriptions"):
        """
        Transcribe video/audio to SRT-format subtitle string.
        If save=True, writes .srt to output_folder.
        """
        if save and not os.path.exists(output_folder):
            os.makedirs(output_folder)

        segments, info = self.model.transcribe(
            video_path,
            language="en",
            beam_size=10,
            best_of=10,
            patience=2.0,
            length_penalty=1.0,
            temperature=[0.0, 0.2],
            word_timestamps=True,
            condition_on_previous_text=True,
            no_repeat_ngram_size=3,
            log_prob_threshold=-2.0,
            suppress_blank=True,
            suppress_tokens=[-1],
            vad_filter=True,
            chunk_length=30,
        )

        srt_output = []
        for i, segment in enumerate(segments, 1):
            start = self._format_timestamp(segment.start)
            end = self._format_timestamp(segment.end)
            text = segment.text.strip()
            srt_output.append(f"{i}\n{start} --> {end}\n{text}\n")

        srt_string = "\n".join(srt_output)

        if save:
            save_index = len(os.listdir(output_folder))
            srt_path = Path(output_folder) / f"transcription_{save_index}.srt"
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_string)

        return srt_string

    def _format_timestamp(self, seconds):
        """Convert float seconds to SRT timestamp HH:MM:SS,mmm."""
        millis = int(seconds * 1000)
        hours = millis // 3600000
        mins = (millis % 3600000) // 60000
        secs = (millis % 60000) // 1000
        ms = millis % 1000
        return f"{hours:02}:{mins:02}:{secs:02},{ms:03}"
