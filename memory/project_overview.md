---
name: Mushroom Disease Detection Project Overview
description: Current state of all phases, model results, and key constraints
type: project
---

Phases 1-3 complete. Phase 4 (evaluation) and Phase 5 (deployment) remaining.

**Dataset:** 761 images, 3 classes — Healthy (299), Single_Infected (147), Mixed_Infected (315). Augmented to 2,400 training images. No pixel-level annotations.

**Phase 3 Results (CPU only):**
- Custom CNN: 85.84% val accuracy (BEST — surprisingly beat all transfer learning)
- InceptionV3: 83.19%
- DenseNet201: 73.45%
- ResNet50: 67.26%
- EfficientNetB0: 44.25%
- VGG16: skipped (too slow on CPU)

**Why transfer learning underperformed:** CPU-only training, insufficient epochs. Reference paper (Wongpanya) got 92.5% with DenseNet201 on GPU.

**Why:** Need to improve beyond 85.84% for a publishable/satisfactory result.

**How to apply:** Recommend GPU retraining on Colab + better architectures (EfficientNetB3/B4). Do NOT attempt image segmentation — no pixel masks exist, too time-intensive to annotate in one session.
