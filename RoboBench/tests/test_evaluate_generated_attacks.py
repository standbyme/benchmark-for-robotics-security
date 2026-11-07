import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluate_generated_attacks as evaluator  # noqa: E402
from robobench.index import Index  # noqa: E402


def test_only_ready_attack_is_evaluated(monkeypatch, tmp_path):
    (tmp_path / "README.md").write_text("- config_name: DROID\n- config_name: shawshank_DROID\n")
    index = Index([{"id": "1", "dataset": "DROID"},
                   {"id": "2", "dataset": "shawshank_DROID"}])
    monkeypatch.setattr(evaluator, "TASKS", [("shawshank", "DROID"), ("shawshank", "RH20T")])
    monkeypatch.setattr(evaluator, "load_dotenv", lambda: None)
    monkeypatch.setattr(evaluator, "load_index", lambda _: index)
    monkeypatch.setattr(evaluator, "make_defense", lambda *args: "defense")
    scenes_seen, cells, reports = [], [], []
    monkeypatch.setattr(evaluator, "scene_frame", lambda name, kind, repo:
                        (scenes_seen.append((name, kind)) or ("frame", "scenes")))
    monkeypatch.setattr(evaluator, "run_cell", lambda *args: cells.append(args))
    monkeypatch.setattr(evaluator, "score_experiment", lambda *args: reports.append(args))
    evaluator.main(tmp_path, "model", 2, False, 1)
    assert scenes_seen == [("DROID", "benign"), ("shawshank_DROID", "malicious")]
    assert len(cells) == 2 and len(reports) == 1
