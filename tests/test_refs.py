from pathlib import Path

from refs import import_refs, list_refs, resolve_ref_path


def test_list_refs_filters_extensions(tmp_path):
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()
    (ref_dir / "voice.wav").write_text("ok")
    (ref_dir / "skip.txt").write_text("no")
    refs = list_refs(ref_dir)
    assert refs == ["voice.wav"]


def test_import_refs_handles_collisions(tmp_path):
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()
    source = tmp_path / "source.wav"
    source.write_text("audio")

    first = import_refs([source], directory=ref_dir)
    assert len(first) == 1
    assert (ref_dir / first[0]).exists()

    second = import_refs([source], directory=ref_dir)
    assert len(second) == 1
    assert (ref_dir / second[0]).exists()
    assert first[0] != second[0]


def test_resolve_ref_path(tmp_path):
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()
    sample = ref_dir / "voice.wav"
    sample.write_text("audio")
    path = resolve_ref_path("voice.wav", directory=ref_dir)
    assert Path(path) == sample
