"""Download recording from URL and normalize to MP3 for AssemblyAI."""

import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

# Inputs the API accepts (extension including dot, lowercase).
ALLOWED_MEDIA_FORMATS = frozenset(
    {
        ".wav",
        ".mp3",
        ".mp4",
        ".mpeg",
        ".mov",
        ".webm",
        ".flac",
        ".ogg",
        ".mkv",
        ".3gp",
    }
)

# Major audio/* and video/* MIME types from clients → extension for temp file + ffmpeg.
_MIME_TO_MEDIA_FORMAT: dict[str, str] = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/x-matroska": ".mkv",
    "video/3gpp": ".3gp",
    "audio/mp4": ".mp4",
    "audio/x-m4a": ".mp4",
    "audio/m4a": ".mp4",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/3gpp": ".3gp",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mpeg3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/webm": ".webm",
    "video/quicktime": ".mov",
    "video/mpeg": ".mpeg",
}


def normalize_media_format(media_format: str) -> str:
    s = (media_format or "").strip().lower()
    if not s.startswith("."):
        s = f".{s}"
    return s


def validate_signed_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("https", "http"):
        raise ValueError("signed_url must be http or https")


def validate_media_format(ext: str) -> None:
    if ext not in ALLOWED_MEDIA_FORMATS:
        allowed = ", ".join(sorted(ALLOWED_MEDIA_FORMATS))
        raise ValueError(f"Resolved file type must be one of: {allowed}")


def extension_from_content_type(content_type: str) -> str:
    """
    Map Content-Type (MIME) to a concrete extension for download + ffmpeg.
    Only ``audio/*`` and ``video/*`` types are accepted; subtype must be in the supported map.
    """
    raw = (content_type or "").strip()
    if not raw:
        raise ValueError("contentType must not be empty")

    mime = raw.split(";")[0].strip().lower()
    if not (mime.startswith("audio/") or mime.startswith("video/")):
        raise ValueError(
            f"contentType must start with audio/ or video/, got {content_type!r}"
        )

    ext = _MIME_TO_MEDIA_FORMAT.get(mime)
    if ext is None:
        known = ", ".join(sorted(_MIME_TO_MEDIA_FORMAT))
        raise ValueError(
            f"Unsupported contentType {content_type!r}. Supported MIME types: {known}"
        )
    validate_media_format(ext)
    return ext


def assert_ffmpeg_available() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required on PATH to normalize audio for AssemblyAI")


class NoAudioStreamError(Exception):
    """The file downloaded and opened fine, but carries no audio track.

    Deliberately not a ``RuntimeError``: the route has to tell "this recording can never produce a
    transcript" apart from "the transcode broke". The first is a permanent property of the
    recording and no retry will change it; the second is worth another attempt.
    """


def has_audio_stream(path: Path) -> bool:
    """Whether ffprobe can see at least one audio stream in the file.

    Returns ``True`` whenever the answer cannot be established — ffprobe absent, or a file it
    refuses to read at all. Those carry on to ffmpeg and fail there exactly as they do today, so
    this check can only ever add the no-audio outcome; it can never take an existing one away.
    """
    if not shutil.which("ffprobe"):
        return True
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return True
    return bool(proc.stdout.strip())


# Stream body to disk so peak RAM does not scale with file size (long recordings).
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024


def download_to_path(url: str, dest: Path, timeout_s: float = 1800.0) -> None:
    validate_signed_url(url)
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        with client.stream("GET", url) as r:
            r.raise_for_status()
            with dest.open("wb") as f:
                for chunk in r.iter_bytes(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                    f.write(chunk)


def transcode_to_mp3(input_path: Path, output_mp3: Path) -> None:
    """Extract/encode audio to MP3 (AssemblyAI-friendly). Requires ffmpeg."""
    assert_ffmpeg_available()
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "libmp3lame",
        "-q:a",
        "4",
        str(output_mp3),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({proc.returncode}): {proc.stderr or proc.stdout or 'no stderr'}"
        )


def prepare_assembly_audio_mp3_paths(signed_url: str, media_format: str) -> tuple[Path, Path]:
    """Download and transcode to MP3. Returns (mp3_path, work_dir); caller deletes work_dir."""
    ext = normalize_media_format(media_format)
    validate_media_format(ext)

    tmpdir = tempfile.mkdtemp(prefix="qlu_transcribe_")
    base = Path(tmpdir)
    raw_path = base / f"input{ext}"
    mp3_path = base / "normalized.mp3"
    try:
        t_dl0 = time.perf_counter()
        download_to_path(signed_url, raw_path)
        t_dl1 = time.perf_counter()
        print(f"Downloading took {t_dl1 - t_dl0:.2f} seconds", flush=True)
        # Asked before transcoding rather than inferred from ffmpeg's failure afterwards. The
        # transcode strips video (``-vn``), so a file with no audio track leaves ffmpeg with no
        # streams to write and it fails with a message about the OUTPUT — which reads as a
        # transcode fault and gets reported as one, when the recording simply had no sound in it.
        if not has_audio_stream(raw_path):
            raise NoAudioStreamError(
                "Recording contains no audio track; there is nothing to transcribe."
            )
        t_pp0 = time.perf_counter()
        transcode_to_mp3(raw_path, mp3_path)
        t_pp1 = time.perf_counter()
        print(f"Preprocessing took {t_pp1 - t_pp0:.2f} seconds", flush=True)
        return mp3_path, base
    except Exception:
        shutil.rmtree(base, ignore_errors=True)
        raise
