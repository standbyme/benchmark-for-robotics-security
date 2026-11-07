import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import finalize_generated_attacks as finalize  # noqa: E402


def test_finalizer_requires_the_complete_source_key_set(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    entries = [{"id": f"id-{i}", "dataset": "DROID", "key": f"DROID/f{i}.png"}
               for i in range(2)]
    (artifacts / "index.json").write_text(json.dumps(entries))
    tree = tmp_path / "trees/shawshank_DROID"
    tree.mkdir(parents=True)
    instructions = tree / "DROID_instr.json"
    instructions.write_text(json.dumps({"dataset/f0.png": ["m", "b"]}))
    built = []
    monkeypatch.setattr(finalize, "TASKS", [("shawshank", "DROID")])
    monkeypatch.setattr(finalize, "build", lambda *args: built.append(args))
    finalize.main(artifacts, tree.parent)
    assert not built
    instructions.write_text(json.dumps({"dataset/f0.png": ["m", "b"], "dataset/f1.png": ["m", "b"]}))
    finalize.main(artifacts, tree.parent)
    assert len(built) == 1
    instructions.write_text(json.dumps({"dataset/unknown.png": ["m", "b"]}))
    with pytest.raises(ValueError, match="not in source artifact"):
        finalize.main(artifacts, tree.parent)
