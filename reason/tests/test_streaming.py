"""Verified against the table in
docs/reasoningModel/01-gemma-reasoning-mode.md §9. No model, no GPU.
"""
from reason.src.runtime.streaming import SentenceAssembler


def run(deltas, **kwargs):
    assembler = SentenceAssembler(**kwargs)
    out = []
    for delta in deltas:
        out += assembler.push(delta)
    out += assembler.flush()
    return out


def test_multi_sentence_token_by_token():
    out = run(["Hello", " there", ". ", "How can", " I help", "?"])
    assert out == ["Hello there.", "How can I help?"]


def test_abbreviation_is_not_a_sentence_end():
    out = run(["Dr. Smith is here. ", "Next one."])
    assert out == ["Dr. Smith is here.", "Next one."]


def test_decimal_is_not_a_sentence_end():
    out = run(["It costs 3.50 dollars. Ok."])
    assert out == ["It costs 3.50 dollars.", "Ok."]


def test_quote_after_terminator():
    out = run(['He said "hi." ', "Then left."])
    assert out == ['He said "hi."', "Then left."]


def test_multiple_ends_in_one_delta():
    out = run(["One. Two. Three."])
    assert out == ["One.", "Two.", "Three."]


def test_never_resolved_abbreviation_flushes_at_end():
    out = run(["Ask Dr."])
    assert out == ["Ask Dr."]


def test_no_terminator_falls_back_to_max_chars():
    out = run(["word " * 60], max_chars=40)
    assert len(out) > 1
    assert all(len(s) <= 40 for s in out[:-1])


def test_empty_input_yields_nothing():
    assert run([]) == []


def test_reply_ending_exactly_on_boundary_has_no_empty_tail():
    out = run(["Done. "])
    assert out == ["Done."]


def test_ellipsis():
    out = run(["Hmm... Maybe not."])
    assert out == ["Hmm...", "Maybe not."]


def test_state_does_not_leak_across_instances():
    a = SentenceAssembler()
    a.push("Partial sentence without end")
    b = SentenceAssembler()
    assert b.push("Complete.") == ["Complete."]
    assert b.flush() == []


def test_devanagari_danda_splits_hindi_sentences():
    # Reason mode's default assistant replies in Hindi; Piper needs each
    # sentence as soon as the danda lands, not only at flush/max_chars.
    out = run(["नमस्ते। ", "मैं मदद के लिए तैयार हूँ।"])
    assert out == ["नमस्ते।", "मैं मदद के लिए तैयार हूँ।"]
