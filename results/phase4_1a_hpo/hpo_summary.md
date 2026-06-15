# Phase 4.1a — HPO Sweep Summary

**Date:** 2026-06-15 12:09
**Fold:** fold_0 (validate on fold 0, train on folds 1-4)
**Dataset:** C-NMC 2019 (train=8817, val=1844)
**Total runs:** 51 (51 complete, 0 failed)
**Total GPU-hours:** 22.93h
**Wall-clock time:** 20.42h

## Selected Learning Rates

| Model | Type | Selected LR | Val BalAcc | Best Ep | Batch | VRAM (MB) | GPU-h |
|-------|------|------------|-----------|---------|-------|-----------|-------|
| convnext_t | FT | 5e-04 | 0.9415 | 44 | 32 | 4241 | 0.418 |
| dinobloom_base_ft | FT | 5e-05 | 0.9374 | 41 | 32 | 10805 | 1.085 |
| dinobloom_base_lp | LP | 1e-03 | 0.8189 | 27 | 32 | 10809 | 0.292 |
| dinobloom_small_ft | FT | 5e-05 | 0.9236 | 14 | 32 | 5520 | 0.247 |
| dinobloom_small_lp | LP | 5e-04 | 0.8120 | 16 | 32 | 10809 | 0.118 |
| edgenext_s | FT | 5e-04 | 0.9329 | 49 | 32 | 2088 | 0.417 |
| edgenext_xs | FT | 5e-04 | 0.9302 | 41 | 32 | 4669 | 0.332 |
| efficientformer_l1 | FT | 1e-04 | 0.9373 | 77 | 32 | 3503 | 0.482 |
| efficientnet_b0 | FT | 5e-04 | 0.9273 | 57 | 32 | 3503 | 0.434 |
| fastvit_t12 | FT | 5e-04 | 0.9407 | 68 | 32 | 2866 | 0.666 |
| fastvit_t8 | FT | 5e-04 | 0.9412 | 51 | 32 | 2118 | 0.417 |
| mobilevit_s | FT | 5e-04 | 0.9371 | 32 | 32 | 4665 | 0.327 |
| mobilevit_xs | FT | 1e-04 | 0.9209 | 62 | 32 | 3756 | 0.501 |
| resnet50 | FT | 5e-04 | 0.9189 | 41 | 32 | 3503 | 0.340 |
| tinyvit_11m | FT | 1e-04 | 0.9323 | 67 | 32 | 3503 | 0.547 |
| tinyvit_5m | FT | 5e-04 | 0.9471 | 62 | 32 | 3115 | 0.491 |
| vit_b16 | FT | 1e-04 | 0.9200 | 51 | 32 | 5519 | 0.749 |

## LR Comparison (all candidates)

| Model | LR=5e-4 | LR=1e-4 | LR=5e-5 | Selected |
|-------|---------|---------|---------|----------|
| mobilevit_xs | 0.9143 | 0.9209 | 0.9024 | **1e-04** |
| mobilevit_s | 0.9371 | 0.9085 | 0.9185 | **5e-04** |
| edgenext_xs | 0.9302 | 0.9241 | 0.9115 | **5e-04** |
| edgenext_s | 0.9329 | 0.9305 | 0.9202 | **5e-04** |
| fastvit_t8 | 0.9412 | 0.9138 | 0.9030 | **5e-04** |
| fastvit_t12 | 0.9407 | 0.9306 | 0.9214 | **5e-04** |
| tinyvit_5m | 0.9471 | 0.8955 | 0.8822 | **5e-04** |
| tinyvit_11m | 0.9278 | 0.9323 | 0.9077 | **1e-04** |
| efficientformer_l1 | 0.9306 | 0.9373 | 0.9230 | **1e-04** |
| efficientnet_b0 | 0.9273 | 0.9056 | 0.8713 | **5e-04** |
| resnet50 | 0.9189 | 0.8764 | 0.8626 | **5e-04** |
| convnext_t | 0.9415 | 0.9340 | 0.9288 | **5e-04** |
| vit_b16 | 0.8952 | 0.9200 | 0.9161 | **1e-04** |

### DinoBloom Fine-tuning

| Model | LR=5e-5 | LR=1e-5 | LR=5e-6 | Selected |
|-------|---------|---------|---------|----------|
| dinobloom_small_ft | 0.9236 | 0.9133 | 0.9167 | **5e-05** |
| dinobloom_base_ft | 0.9374 | 0.9189 | 0.9166 | **5e-05** |

### DinoBloom Linear Probing

| Model | LR=1e-3 | LR=5e-4 | LR=1e-4 | Selected |
|-------|---------|---------|---------|----------|
| dinobloom_small_lp | 0.8058 | 0.8120 | 0.8098 | **5e-04** |
| dinobloom_base_lp | 0.8189 | 0.8161 | 0.8170 | **1e-03** |
