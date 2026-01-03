from text_tools import chunk_script


def test_chunker_newline_split():
    text = (
        "Premiere ligne assez longue pour durer un peu\n"
        "Deuxieme ligne avec plusieurs mots pour eviter merge\n"
        "Troisieme ligne encore plus longue pour stabiliser"
    )
    chunks = chunk_script(text, min_words_per_chunk=2, max_words_without_terminator=40)
    assert len(chunks) == 3
    assert chunks[0].reason == "newline"


def test_chunker_disable_newline_split():
    text = "Une ligne courte mais correcte\nUne autre ligne pour tester"
    chunks = chunk_script(
        text,
        min_words_per_chunk=2,
        max_words_without_terminator=40,
        split_on_newline=False,
    )
    assert len(chunks) == 1


def test_chunker_min_words_blocks_split():
    text = "Bonjour\nMerci beaucoup"
    chunks = chunk_script(
        text,
        min_words_per_chunk=16,
        max_words_without_terminator=40,
    )
    assert len(chunks) == 1
    assert "newline_boundary_skipped_min_words" in chunks[0].warnings


def test_chunker_terminator_prevents_mid_sentence_split():
    text = "Mot mot mot mot mot. Suite suite suite suite suite."
    chunks = chunk_script(
        text,
        min_words_per_chunk=1,
        max_words_without_terminator=50,
        max_est_seconds_per_chunk=2.5,
    )
    assert len(chunks) >= 2
    assert chunks[0].reason == "terminator"


def test_chunker_fallback_order():
    text = "Un bloc long: suite longue; encore long — suite encore, encore encore fin"
    chunks = chunk_script(
        text,
        min_words_per_chunk=1,
        max_words_without_terminator=3,
        max_est_seconds_per_chunk=20.0,
    )
    reasons = [chunk.reason for chunk in chunks]
    assert any(reason.startswith("fallback(:)") for reason in reasons)
    assert any(reason.startswith("fallback(;)") for reason in reasons)
    assert any(reason.startswith("fallback(—)") for reason in reasons)
    assert any(reason.startswith("fallback(,)") for reason in reasons)


def test_chunker_hard_split_no_punct():
    text = "Mot mot mot mot mot mot mot mot mot"
    chunks = chunk_script(
        text,
        min_words_per_chunk=1,
        max_words_without_terminator=3,
        max_est_seconds_per_chunk=20.0,
    )
    assert any(chunk.reason == "hard" for chunk in chunks)
    assert any("hard_split_no_punct" in chunk.warnings for chunk in chunks)


def test_end_chunk_min_words_enforced_after_fallback():
    text = "Un deux trois: fin"
    chunks = chunk_script(
        text,
        min_words_per_chunk=3,
        max_words_without_terminator=3,
        max_est_seconds_per_chunk=20.0,
    )
    assert len(chunks) == 1
    assert chunks[0].word_count >= 3


def test_min_words_clamped_to_20():
    text = " ".join(["Mot"] * 21) + "\nfin fin"
    chunks = chunk_script(
        text,
        min_words_per_chunk=25,
        max_words_without_terminator=40,
        max_est_seconds_per_chunk=20.0,
    )
    assert len(chunks) == 2
    assert chunks[0].word_count == 21

