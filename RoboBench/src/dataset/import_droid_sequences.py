"""Import annotated, ordered windows from the pinned LeRobot DROID v3 files."""

import json
from pathlib import Path

import av
import pandas as pd
import typer

CAMERA = "observation.images.exterior_image_1_left"
REVISION = "87301a2d2e99340e2010c9ef0f1d8e780b08aaf9"


def main(source: Path, annotations: Path, output: Path):
    """Read locally downloaded DROID files and explicit annotations into a raw tree.

    Each annotation has episode, offsets (episode-relative frame indices),
    malicious and benign. Offsets must be strictly increasing and within the
    episode. The output directory name becomes the artifact name.
    """
    info = json.loads((source / "meta/info.json").read_text())
    fps = info["fps"]
    episodes = pd.read_parquet(source / "meta/episodes/chunk-000/file-000.parquet")
    episodes = episodes.set_index("episode_index")
    if not episodes.index.is_unique:
        raise ValueError("episode metadata contains duplicate episode indices")
    records = json.loads(annotations.read_text())
    if not records:
        raise ValueError("annotations are empty")
    instructions, provenance = {}, []
    seen = set()
    for item in records:
        episode = item["episode"]
        offsets = item["offsets"]
        if episode in seen:
            raise ValueError(f"duplicate episode {episode}")
        seen.add(episode)
        if not offsets or any(type(i) is not int for i in offsets) or sorted(set(offsets)) != offsets:
            raise ValueError(f"episode {episode}: offsets must be strictly increasing integers")
        row = episodes.loc[episode]
        if offsets[0] < 0 or offsets[-1] >= row["length"]:
            raise ValueError(f"episode {episode}: frame offset outside episode")
        if not all(isinstance(item.get(k), str) and item[k].strip() for k in ("malicious", "benign")):
            raise ValueError(f"episode {episode}: both instructions are required")
        prefix = f"videos/{CAMERA}"
        start = round(float(row[f"{prefix}/from_timestamp"]) * fps)
        chunk = int(row[f"{prefix}/chunk_index"])
        file_index = int(row[f"{prefix}/file_index"])
        video = source / "videos" / CAMERA / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
        desired = {start + offset: offset for offset in offsets}
        paths = {}
        with av.open(str(video)) as container:
            for position, frame in enumerate(container.decode(video=0)):
                if position in desired:
                    offset = desired[position]
                    path = output / "dataset" / f"episode_{episode:06d}_frame_{offset:06d}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    frame.to_image().save(path)
                    paths[offset] = path.as_posix()
                if position >= max(desired):
                    break
        if len(paths) != len(offsets):
            raise ValueError(f"episode {episode}: video did not contain all requested frames")
        ordered = [paths[offset] for offset in offsets]
        instructions[ordered[0]] = [item["malicious"], item["benign"], ordered[1:]]
        provenance.append({"episode": episode, "offsets": offsets, "fps": fps,
                           "camera": CAMERA, "video": video.relative_to(source).as_posix(),
                           "source_revision": REVISION, "source_dataset": "lerobot/droid_100"})
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{output.name}_instr.json").write_text(json.dumps(instructions, indent=2))
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    typer.run(main)
