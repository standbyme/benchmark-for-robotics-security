---
configs:
- config_name: DROID
  data_files:
  - split: train
    path: data/DROID/train.parquet
- config_name: RH20T
  data_files:
  - split: train
    path: data/RH20T/train.parquet
- config_name: RJB-Instructions
  data_files:
  - split: train
    path: data/RJB-Instructions/train.parquet
- config_name: RJB-Instructions-5
  data_files:
  - split: train
    path: data/RJB-Instructions-5/train.parquet
- config_name: ROBOVQA
  data_files:
  - split: train
    path: data/ROBOVQA/train.parquet
- config_name: badrobot_cd_DROID
  data_files:
  - split: train
    path: data/badrobot_cd_DROID/train.parquet
- config_name: badrobot_cd_RH20T
  data_files:
  - split: train
    path: data/badrobot_cd_RH20T/train.parquet
- config_name: badrobot_cd_RJB-Instructions-5
  data_files:
  - split: train
    path: data/badrobot_cd_RJB-Instructions-5/train.parquet
- config_name: badrobot_cd_ROBOVQA
  data_files:
  - split: train
    path: data/badrobot_cd_ROBOVQA/train.parquet
- config_name: badrobot_cd_car
  data_files:
  - split: train
    path: data/badrobot_cd_car/train.parquet
- config_name: badrobot_cd_robo2vlm
  data_files:
  - split: train
    path: data/badrobot_cd_robo2vlm/train.parquet
- config_name: badrobot_cj_DROID
  data_files:
  - split: train
    path: data/badrobot_cj_DROID/train.parquet
- config_name: badrobot_cj_RH20T
  data_files:
  - split: train
    path: data/badrobot_cj_RH20T/train.parquet
- config_name: badrobot_cj_RJB-Instructions-5
  data_files:
  - split: train
    path: data/badrobot_cj_RJB-Instructions-5/train.parquet
- config_name: badrobot_cj_ROBOVQA
  data_files:
  - split: train
    path: data/badrobot_cj_ROBOVQA/train.parquet
- config_name: badrobot_cj_car
  data_files:
  - split: train
    path: data/badrobot_cj_car/train.parquet
- config_name: badrobot_cj_robo2vlm
  data_files:
  - split: train
    path: data/badrobot_cj_robo2vlm/train.parquet
- config_name: badrobot_sm_DROID
  data_files:
  - split: train
    path: data/badrobot_sm_DROID/train.parquet
- config_name: badrobot_sm_RH20T
  data_files:
  - split: train
    path: data/badrobot_sm_RH20T/train.parquet
- config_name: badrobot_sm_RJB-Instructions-5
  data_files:
  - split: train
    path: data/badrobot_sm_RJB-Instructions-5/train.parquet
- config_name: badrobot_sm_ROBOVQA
  data_files:
  - split: train
    path: data/badrobot_sm_ROBOVQA/train.parquet
- config_name: badrobot_sm_car
  data_files:
  - split: train
    path: data/badrobot_sm_car/train.parquet
- config_name: badrobot_sm_robo2vlm
  data_files:
  - split: train
    path: data/badrobot_sm_robo2vlm/train.parquet
- config_name: car
  data_files:
  - split: train
    path: data/car/train.parquet
- config_name: robo2vlm
  data_files:
  - split: train
    path: data/robo2vlm/train.parquet
- config_name: robopair_DROID
  data_files:
  - split: train
    path: data/robopair_DROID/train.parquet
- config_name: robopair_RH20T
  data_files:
  - split: train
    path: data/robopair_RH20T/train.parquet
- config_name: robopair_RJB-Instructions-5
  data_files:
  - split: train
    path: data/robopair_RJB-Instructions-5/train.parquet
- config_name: robopair_ROBOVQA
  data_files:
  - split: train
    path: data/robopair_ROBOVQA/train.parquet
- config_name: robopair_car
  data_files:
  - split: train
    path: data/robopair_car/train.parquet
- config_name: robopair_robo2vlm
  data_files:
  - split: train
    path: data/robopair_robo2vlm/train.parquet
- config_name: shawshank_ROBOVQA
  data_files:
  - split: train
    path: data/shawshank_ROBOVQA/train.parquet
- config_name: shawshank_DROID
  data_files:
  - split: train
    path: data/shawshank_DROID/train.parquet
- config_name: shawshank_RH20T
  data_files:
  - split: train
    path: data/shawshank_RH20T/train.parquet
- config_name: shawshank_robo2vlm
  data_files:
  - split: train
    path: data/shawshank_robo2vlm/train.parquet
- config_name: shawshank_car
  data_files:
  - split: train
    path: data/shawshank_car/train.parquet
- config_name: shawshank_RJB-Instructions-5
  data_files:
  - split: train
    path: data/shawshank_RJB-Instructions-5/train.parquet
---

# RoboBench

Benchmark artifacts for jailbreak attacks on vision-language robot planners.
One config per artifact; every artifact is self-contained (images included).
Vanilla artifacts carry the original malicious and benign instruction per
frame; `<attack>_<base>` artifacts carry the attack's rewritten malicious
instruction with the same frames and benign instruction.

`RJB-Instructions-5` is the 90-scene RJB-Instructions set evaluated in the
paper (5 scenes per taxonomy case). `RJB-Instructions` is its 900-scene
expansion (50 per case) with regenerated images and instructions.

Columns: `id` (uuid, unique across the repo), `dataset`, `category`,
`images`, `benign`, `malicious`, `target`.
