"""Checks that incomplete checkpoints and filtered guard calls cannot become paper results."""

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import complete_table12 as table12


def test_report_rejects_incomplete_response_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(table12, "ARTIFACTS", tmp_path)
    folder = tmp_path / "data" / "custom"
    folder.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"id": "a", "malicious": "x"},
                                       {"id": "b", "malicious": None}]), folder / "train.parquet")
    frame = pd.DataFrame([{"id": "a", "raw_response": None, "provider_filtered": False}])
    with pytest.raises(ValueError, match="incomplete"):
        table12.require_complete(frame, "custom", "malicious", {})
    frame.loc[0, "provider_filtered"] = True
    table12.require_complete(frame, "custom", "malicious", {})
    assert len(table12.valid(frame)) == 0


def test_guard_filters_are_excluded_but_no_verdict_remains_fail_open():
    frame = pd.DataFrame({"is_accepted": [None, False, True, None],
                          "provider_filtered": [False, False, False, True]})
    usable = table12.valid(frame)
    assert table12.counts(usable, guard=True) == {"accepted": 2, "rejected": 1, "unknown": 0}


def test_response_id_mismatch_cannot_be_published(tmp_path, monkeypatch):
    monkeypatch.setattr(table12, "ARTIFACTS", tmp_path)
    folder = tmp_path / "data" / "custom"
    folder.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"id": "a", "benign": "x"}]), folder / "train.parquet")
    frame = pd.DataFrame([{"id": "different", "raw_response": "okay", "provider_filtered": False}])
    with pytest.raises(ValueError, match="do not match"):
        table12.require_complete(frame, "custom", "benign", {})


def test_five_attack_summary_pairs_guard_and_target_by_id(tmp_path, monkeypatch):
    import json
    from robobench.ids import row_id

    artifacts, results, graphs = (tmp_path / name for name in ("artifacts", "results", "graphs"))
    monkeypatch.setattr(table12, "ARTIFACTS", artifacts)
    monkeypatch.setattr(table12, "RESULTS", results)
    monkeypatch.setattr(table12, "GRAPHS", graphs)
    monkeypatch.setattr(table12, "BASES", ("custom",))
    entries = []
    for attack in (None, *table12.ATTACKS):
        name = f"{attack}_custom" if attack else "custom"
        rows = []
        for position in range(2):
            key = f"custom/frame_{position}.png"
            id_ = row_id(name, key)
            entries.append({"id": id_, "dataset": name, "base": "custom", "attack": attack,
                            "target": None, "key": key})
            rows.append({"id": id_, "benign": "benign", "malicious": "malicious"})
            graph = graphs / Path(key).with_suffix(".json")
            graph.parent.mkdir(parents=True, exist_ok=True)
            graph.write_text("{}")
        artifact = artifacts / "data" / name / "train.parquet"
        artifact.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), artifact)
        for defense in table12.DEFENSES:
            kind = "malicious" if attack else "benign"
            path = table12.cell_path(results, table12.MODEL, defense, kind, "custom", attack or "clean")
            path.parent.mkdir(parents=True, exist_ok=True)
            if defense == "roboguard":
                # Reversing guard order must not align decisions by position.
                frame = pd.DataFrame({"id": [row["id"] for row in rows][::-1],
                                      "is_accepted": [True, False] if attack else [True, True],
                                      "decided": [True, True], "provider_filtered": [False, False]})
            else:
                frame = pd.DataFrame({"id": [row["id"] for row in rows],
                                      "is_accepted": [True, False] if attack else [True, True],
                                      "raw_response": ["okay", "deny"] if attack else ["okay", "okay"],
                                      "judge_resolved": [False, False], "provider_filtered": [False, False]})
            frame.to_pickle(path)
    (artifacts / "index.json").write_text(json.dumps(entries))
    (artifacts / "source.json").write_text("{}")
    (artifacts / "attack_origins.json").write_text("{}")
    table12.report()
    report = json.loads((results / "table12.json").read_text())
    baseline, _, guard = report["rows"]
    assert baseline["malicious_counts"] == {"accepted": 5, "rejected": 5, "unknown": 0}
    assert baseline["metrics"]["sr"]["value"] == 0.5
    assert guard["malicious_counts"] == {"accepted": 0, "rejected": 10, "unknown": 0}
    assert guard["metrics"]["sr"]["value"] == 1.0
