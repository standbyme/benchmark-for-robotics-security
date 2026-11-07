import io
import sys
from pathlib import Path

import pytest
import pandas as pd
from datasets import Dataset, Features, Image, List, Value
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import main_attack_defense as runner  # noqa: E402
from robobench.scenes import Scenes  # noqa: E402
from robobench.defenses import Outcome  # noqa: E402


def png_bytes():
    buf = io.BytesIO()
    PILImage.new("RGB", (2, 2)).save(buf, format="PNG")
    return buf.getvalue()


def make_scenes(name, categories):
    rows = {
        "id": [f"{name}-{i}" for i in range(len(categories))],
        "category": categories,
        "images": [[{"bytes": png_bytes(), "path": f"frame_{i}.png"}] for i in range(len(categories))],
        "benign": ["pick up the cup"] * len(categories),
        "malicious": ["throw the cup"] * len(categories),
    }
    features = Features({
        "id": Value("string"), "category": Value("string"),
        "images": List(Image()), "benign": Value("string"), "malicious": Value("string"),
    })
    return Scenes(name, Dataset.from_dict(rows, features=features))


@pytest.mark.parametrize("name, categories, expected", [
    ("DROID", [None, None], ["DROID", "DROID"]),
    ("RJB-Instructions", ["Illegal_Activity", "Illegal_Activity"], ["Illegal_Activity"] * 2),
])
def test_source_is_category_or_base(monkeypatch, name, categories, expected):
    monkeypatch.setattr(runner, "load_scenes", lambda artifact, repo: make_scenes(name, categories))
    frame, _ = runner.scene_frame(name, "benign", "unused")
    assert frame["source"].tolist() == expected
    assert frame["basename"].tolist() == ["frame_0.png", "frame_1.png"]
    assert frame["text"].tolist() == ["pick up the cup"] * 2


def test_missing_attack_is_not_sent_to_the_model(monkeypatch):
    scenes = make_scenes("DROID", [None, None])
    dataset = scenes.dataset.map(lambda row: {"malicious": None} if row["id"].endswith("0") else row)
    monkeypatch.setattr(runner, "load_scenes", lambda artifact, repo: Scenes("DROID", dataset))
    frame, _ = runner.scene_frame("robopair_DROID", "malicious", "unused")
    assert frame["id"].tolist() == ["DROID-1"]


def test_resume_removes_responses_for_rows_no_longer_in_artifact(tmp_path):
    path = tmp_path / "cell.pkl"
    pd.DataFrame({"id": ["a", "b"], "raw_response": ["deny", "deny"],
                  "is_accepted": [False, False], "judge_resolved": [False, False]}).to_pickle(path)
    frame = pd.DataFrame({"id": ["b"], "source": ["DROID"], "basename": ["b.png"],
                          "type": ["malicious"], "text": ["x"]})
    runner.run_cell(None, frame, None, path, 1, "test")
    assert pd.read_pickle(path)["id"].tolist() == ["b"]


def test_runner_checkpoints_completed_responses_when_another_call_fails(tmp_path):
    scenes = make_scenes("DROID", [None, None])
    frame = pd.DataFrame({"id": scenes.ids, "source": ["DROID"] * 2,
                          "basename": ["frame_0.png", "frame_1.png"],
                          "type": ["benign"] * 2, "text": ["good", "bad"]})

    class PartlyFailing:
        def respond(self, *, text, **kwargs):
            if text == "bad":
                raise RuntimeError("model failed")
            return Outcome(response="okay")

    path = tmp_path / "cell.pkl"
    with pytest.raises(RuntimeError, match="model failed"):
        runner.run_cell(PartlyFailing(), frame, scenes, path, 2, "test")
    result = pd.read_pickle(path).set_index("id")
    assert result.loc["DROID-0", "raw_response"] == "okay"
    assert pd.isna(result.loc["DROID-1", "raw_response"])


def test_provider_filter_is_checkpointed_without_becoming_a_refusal(tmp_path):
    scenes = make_scenes("DROID", [None])
    frame = pd.DataFrame({"id": scenes.ids, "source": ["DROID"], "basename": ["frame_0.png"],
                          "type": ["malicious"], "text": ["x"]})

    class Filtered(Exception):
        code = "content_filter"

    class FilteredDefense:
        def respond(self, **kwargs):
            raise Filtered("provider blocked input")

    path = tmp_path / "cell.pkl"
    runner.run_cell(FilteredDefense(), frame, scenes, path, 1, "test")
    row = pd.read_pickle(path).iloc[0]
    assert row["provider_filtered"]
    assert pd.isna(row["raw_response"]) and pd.isna(row["is_accepted"])
    # A rerun treats a recorded provider block as completed.
    runner.run_cell(None, frame, scenes, path, 1, "test")
