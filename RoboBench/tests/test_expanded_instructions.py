import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prepare_expanded_instructions import select, taxonomy  # noqa: E402


def test_subset_is_balanced_and_stable_under_input_reordering():
    rows = [{"id": f"{c}-{i}", "dataset": "RJB-Instructions",
             "key": f"group/${c}Example {i}.png"}
            for c in range(18) for i in range(1, 51)]
    first = select(rows)
    assert len(first) == 360
    assert set(Counter(taxonomy(row["key"]) for row in first).values()) == {20}
    assert first == select(list(reversed(rows)))


def test_incomplete_source_is_rejected():
    with pytest.raises(ValueError, match="Expected 18"):
        select([])
