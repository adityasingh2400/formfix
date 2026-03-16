# Datasets And References

This file focuses on the real dataset situation for building a serious comparison engine.

## Bottom line

There is no single public dataset today that is already perfect for:

- NBA or elite-player shot comparison
- per-shot trimming
- biomechanics phase labels
- player identity
- enough clips per player and shot type
- clean commercial licensing

So the practical stack is:

1. use public sports datasets for tracking, pose, and pretraining
2. use licensed basketball footage for real shot comparison
3. export FormFix features from that footage
4. train archetypes first
5. train player profiles next

## Best current public building blocks

### 1. BASKET

Source:

- https://github.com/yulupan00/BASKET
- https://arxiv.org/abs/2503.20781

Why it matters:

- It is currently the strongest large-scale basketball skill-estimation dataset foundation.
- The project describes **4,477 hours**, **32,232 players**, **20 skill attributes**, and about **1.8TB** of data.
- It is useful for pretraining basketball-specific video representations and athlete-skill embeddings.

Why it is not enough alone:

- It is not a shot-mechanics dataset with phase labels.
- It is not already packaged as a player-by-player shot-form library.

Use in FormFix:

- pretrain basketball video encoders
- learn coarse athlete or style embeddings
- help bootstrap retrieval models before shot-specific data is large enough

### 2. TrackID3x3

Source:

- https://github.com/open-starlab/TrackID3x3

Why it matters:

- It is a basketball-specific **multimodal and multicamera** dataset from a **3x3 basketball tournament**.
- It includes player trajectories, tracklets, and body-orientation information.
- The repo states a **CC BY 4.0** license.

Why it is useful:

- good for multi-camera tracking
- good for pose-plus-tracking research
- useful for learning camera-robust basketball motion features

Why it is not enough alone:

- it is not centered on isolated shot clips with per-phase coaching labels

### 3. DeepSport / Basketball-Instants

Source:

- https://github.com/DeepSportradar/instance-segmentation-challenge

Why it matters:

- The challenge repo is based on the **Basketball-Instants-Dataset**
- It references **223 train images** and **64 test images**
- It is distributed through Kaggle and references **CC BY-NC-ND 4.0**

Why it is useful:

- strong for player/ball detection
- good for occlusion-heavy basketball scenes
- useful for improving visual preprocessing and tracking confidence

Why it is not enough alone:

- image-centric rather than shot-sequence-centric
- too small for a comparison engine by itself

### 4. BasketLiDAR

Source:

- https://sites.google.com/keio.jp/keio-csg/projects/basket-lidar

Why it matters:

- It is one of the more interesting newer basketball tracking datasets because it adds a 3D sensing angle.

Why it is useful:

- can help with tracking and camera-robust motion geometry
- useful if you want stronger 3D supervision later

Why it is not enough alone:

- not a shot-form coaching dataset
- access is more limited than a normal fully open download

### 5. SportsMOT

Source:

- https://deeperaction.github.io/datasets/sportsmot.html

Why it matters:

- not basketball-only, but useful for sports tracking pretraining
- helpful if you want stronger player/ball motion backbones

Why it is not enough alone:

- cross-sport tracking is useful for representation learning, not for final shot-form coaching labels

## Best training-pattern reference

### FineDiving

Source:

- https://github.com/xujinglin/FineDiving

Why it matters:

- It is not basketball, but it is one of the cleanest references for **procedure-aware action quality assessment**
- The project centers around fine-grained scoring with temporal structure

Why it matters for FormFix:

- the comparison engine should eventually be temporal and retrieval-based, not just centroid-based
- FineDiving is a strong example of how to structure a quality-assessment problem where order and timing matter

## What is still missing

To build true player-level shot comparison, you still need your own dataset layer with:

- clean shot windows
- player identity
- shot type
- camera bucket
- optionally distance and make/miss
- enough repeated clips per player

## Recommended acquisition plan

### Short term

- keep using the seeded archetype library in the repo
- export features from any local shot library you collect
- train archetypes from those exported features

### Medium term

- license broadcast or practice footage
- trim clips to shot windows
- attach metadata for player, shot type, camera bucket, and outcome
- run the FormFix export pipeline on every clip

### Long term

- create player profiles from repeated clips
- train a sequence embedding model
- retrieve nearest clips, not just nearest centroids

## Product-safe stance

Do not market the current or near-term public-data path as:

- "compare your shot to any NBA player"

unless you truly have:

- enough clips per player
- the legal right to use them
- a validated player-profile pipeline

The right honest progression is:

- shot archetypes now
- trained archetypes next
- true player matches after a real reference library exists
