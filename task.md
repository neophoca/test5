# Metaphase chromosomes: instance segmentation + 24-class ID

## Goal
Automatically segment and identify human chromosomes in grayscale metaphase images.

- **Input:** 1-channel grayscale metaphase image.
- **Output (per image):** ~46 chromosome instances (23 pairs), each with ID in `{1..22, X, Y}`.

## Model output
For each image, the model must return:
- `boxes`:  `Float32[M, 4]`
- `labels`: `Int64[M]` (1..24)
- `scores`: `Float32[M]`
- `masks`:  `Float32[M, 1, H, W]`

Chromosomes should be paired into 23 groups using predicted IDs.

---

## Datasets used in this project

### Dataset A — Cell Image Library “24 chromosomes” (boxes, 24-class)
- **Source:** https://www.kaggle.com/datasets/snikenb/cell-image-library-for-ultralytics-yolo
- **Type:** metaphase images with 24-class bounding boxes
- **Scale:** 5,000 images; 229,852 chromosomes total
- **Annotations:** box-only; Pascal-VOC XML (converted)

### Dataset D — Cell Image Library “single chromosomes” (boxes)
- **Source:** https://www.kaggle.com/datasets/snikenb/cell-image-library-for-ultralytics-yolo
- **Type:** single-chromosome images/instances (mostly one per image)
- **Annotations:** box-level; Pascal-VOC XML

### Dataset B — AutoKary2022 / Chromosome Instance Segmentation (masks, 24-class)
- **Source:** https://github.com/wangjuncongyu/chromosome-instance-segmentation-dataset
- **Type:** metaphase images with instance masks + 24-class label per chromosome
- **Scale:** 612 images; ~27,000 chromosome instances

---

## Other datasets

### Overlapping/Touching Chromosomes Dataset (overlap masks)
- **Source:** https://data.mendeley.com/datasets/h5b3zbtw8v/1
- **Scale:** 500 overlapping/touching chromosome pairs
- **Annotations:** masks for chromosome #1, chromosome #2, and overlap region

### CRCN-NE Metaphase Dataset (boxes, 24-class)
- **Source:** https://zenodo.org/records/15061709
- **Type:** metaphase images with YOLO-format bounding boxes (24 classes)
- **Scale:** 519 images (+ discarded hard subset)

### BioImLAB single-chromosome classification dataset (24-class)
- **Source:** https://www.kaggle.com/datasets/arifmpthesis/bioimlab-chromosome-data-set-for-classification/data
- **Type:** single chromosome images (classification only)
- **Scale:** 5,474 images; 24 classes

---

## Test sets
- **Primary:** B-test (mask + class)
- **Secondary:** D-test sanity check (count + ID plausibility)

---

## Metrics
- **mAP@0.5**
- **PQ_all** and **mPQ**
- **AJI**

---

## Karyotype sanity metric (lightweight QA)
Given predicted labels:
- expected total count: near 46
- autosomes 1..22 should appear in pairs
- sex chromosomes should match either:
  - **XX:** X=2, Y=0
  - **XY:** X=1, Y=1

Score in `[0, 1]`.
