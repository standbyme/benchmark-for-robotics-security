import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prepare_azure_run import INHERITED, SEQUENCE_INHERITED, main  # noqa: E402


def test_azure_collection_is_isolated_and_preserves_origin(tmp_path):
    source, output = tmp_path / "source", tmp_path / "azure"
    entries = []
    for name in INHERITED:
        path = source / "data" / name
        path.mkdir(parents=True)
        (path / "train.parquet").write_bytes(b"source-data")
        entries.append({"id": name, "dataset": name})
    (source / "index.json").write_text(json.dumps(entries))
    (source / "selection.json").write_text(json.dumps({"source_revision": "pinned"}))
    main(source, output)
    assert json.loads((output / "inherited_artifacts.json").read_text())["source_revision"] == "pinned"
    assert (output / "data/DROID/train.parquet").read_bytes() == b"source-data"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(source, output)


def test_sequence_collection_does_not_inherit_old_visual_attack(tmp_path):
    source, output = tmp_path / "source", tmp_path / "sequence"
    entries = []
    for name in SEQUENCE_INHERITED:
        path = source / "data" / name
        path.mkdir(parents=True)
        (path / "train.parquet").write_bytes(b"frame-data")
        entries.append({"id": name, "dataset": name})
    (source / "index.json").write_text(json.dumps(entries))
    (source / "provenance.json").write_text(json.dumps([{"source_revision": "episode-revision"}]))
    main(source, output, sequence=True)
    assert not (output / "data/shawshank_DROID-Sequence-5").exists()
    assert json.loads((output / "inherited_artifacts.json").read_text())["source_revision"] == "episode-revision"
