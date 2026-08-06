"""Noise / filler filters for reason-mode mic spam."""
from pipeline.realtime.noise import (
    is_noise_transcript,
    pcm_peak_abs,
    pick_speech_lang,
    strip_unspeakable,
)


def test_drops_classic_whisper_fillers():
    for t in ("you", "YOU", "Bye", "mm", "hmm", "hi", "thank you", "Thank you for watching!"):
        assert is_noise_transcript(t), t


def test_keeps_real_requests():
    for t in (
        "explain blockchain in tamil",
        "what is the capital of France",
        "translate this to Hindi please",
    ):
        assert not is_noise_transcript(t), t


def test_drops_empty_and_punctuation_only():
    assert is_noise_transcript("")
    assert is_noise_transcript("   ")
    assert is_noise_transcript("...")


def test_pick_speech_lang_devanagari_vs_latin():
    assert pick_speech_lang("नमस्ते दोस्त") == "hi"
    assert pick_speech_lang("Blockchain oru vithamaanam") == "en"
    assert pick_speech_lang("Hello there") == "en"


def test_strip_unspeakable_removes_emoji():
    assert strip_unspeakable("नमस्ते! 😊") == "नमस्ते!"
    assert "😎" not in strip_unspeakable("ok 😎")


def test_pcm_peak_abs_silence_and_tone():
    assert pcm_peak_abs(b"") == 0
    assert pcm_peak_abs(b"\x00\x00" * 10) == 0
    # little-endian int16 10000
    sample = (10000).to_bytes(2, "little", signed=True)
    assert pcm_peak_abs(sample * 4) == 10000
