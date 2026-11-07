import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from robobench.ids import row_id  # noqa: E402
from score_experiments import main  # noqa: E402


def test_scores_visual_attack_with_local_index_and_unknown_response(tmp_path):
    base = "custom"
    attack = "shawshank_custom"
    key = "custom/frame.png"
    entries = [{"id": row_id(name, key), "dataset": name, "base": base,
                "attack": kind, "target": None, "key": key}
               for name, kind in [(base, None), (attack, "shawshank")]]
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(entries))
    artifact = tmp_path / "data" / attack
    artifact.mkdir(parents=True)
    (artifact / "generation_failures.json").write_text(json.dumps({"source-id": {"reason": "provider_content_filter"}}))
    root = tmp_path / "cells/model"
    benign = root / "benign/custom/clean/no_defense.pkl"
    malicious = root / "malicious/custom/shawshank/no_defense.pkl"
    benign.parent.mkdir(parents=True)
    malicious.parent.mkdir(parents=True)
    pd.DataFrame([{"id": entries[0]["id"], "raw_response": "okay", "is_accepted": True,
                   "judge_resolved": False}]).to_pickle(benign)
    pd.DataFrame([{"id": entries[1]["id"], "raw_response": "unclear", "is_accepted": None,
                   "judge_resolved": False}]).to_pickle(malicious)
    output = tmp_path / "report.json"
    main(base, index_path, output, tmp_path / "cells", "model", "no_defense")
    block = json.loads(output.read_text())["datasets"][attack]
    assert block["coverage"] == {"benign": 1, "malicious": 1}
    assert block["counts"]["malicious"]["unknown"] == 1
    assert json.loads(output.read_text())["meta"]["generation_failures"][attack] == {"provider_content_filter": 1}
    report = json.loads(output.read_text())
    assert report["overall"]["counts"]["benign"]["accepted"] == 1
    assert report["datasets"][attack]["counts"]["benign"]["accepted"] == 1


def test_provider_filtered_responses_are_excluded_and_counted(tmp_path):
    base, attack, key = "custom", "shawshank_custom", "custom/frame.png"
    entries = [{"id": row_id(name, key), "dataset": name, "base": base,
                "attack": kind, "target": None, "key": key}
               for name, kind in [(base, None), (attack, "shawshank")]]
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(entries))
    root = tmp_path / "cells/model"
    benign = root / "benign/custom/clean/no_defense.pkl"
    malicious = root / "malicious/custom/shawshank/no_defense.pkl"
    benign.parent.mkdir(parents=True)
    malicious.parent.mkdir(parents=True)
    pd.DataFrame([{"id": entries[0]["id"], "raw_response": "okay", "is_accepted": True}]).to_pickle(benign)
    pd.DataFrame([{"id": entries[1]["id"], "raw_response": None, "is_accepted": None,
                   "provider_filtered": True}]).to_pickle(malicious)
    output = tmp_path / "report.json"
    main(base, index_path, output, tmp_path / "cells", "model", "no_defense")
    report = json.loads(output.read_text())
    assert report["datasets"][attack]["coverage"]["malicious"] == 0
    assert report["meta"]["provider_filtered"][attack]["malicious"] == 1
