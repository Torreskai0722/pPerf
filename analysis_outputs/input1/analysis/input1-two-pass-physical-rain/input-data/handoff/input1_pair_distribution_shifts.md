# Input1 Model-Pair Distribution Shifts

## Core insight

Adverse-input latency sensitivity is pair-conditioned. Different model pairs
shift different parts of the inference-time distribution: some move the
center, some move the lower quartile, and others primarily change dispersion
or center-to-tail separation. Pair identity and MPS mode therefore provide
more useful predictability information than isolated per-model latency alone.

## Metrics and working thresholds

- `CV = standard deviation / mean`
- `tail ratio = (P99 - P50) / P50`
- Large CV or tail-ratio change: at least 25% relative to the matching clear
  condition.
- Large P25, P50, or P99 change: at least 10% relative to the matching clear
  condition.

All comparisons use the clear run for the same model pair and MPS mode.

## Largest changes

| Metric | Non-MPS | MPS |
|---|---|---|
| CV | **PointPillars + YOLOv3:** PointPillars +795% at 7.5 mm/h. **PointPillars + DINO:** PointPillars -54% to -76%. **3DSSD + DeepLabV3+:** 3DSSD -30% to -34%. | **3DSSD + Mask R-CNN:** Mask R-CNN +26% to +38% across all rain levels. **PointPillars + Mask R-CNN:** Mask R-CNN +25% to +36%. **PointPillars + YOLOv3:** YOLOv3 +26% to +28%. |
| `(P99-P50)/P50` | **CenterPoint + DINO:** CenterPoint +190% to +223% across all rain levels. **3DSSD + Faster R-CNN:** Faster R-CNN up to +70%. **CenterPoint + Mask R-CNN:** CenterPoint -44% to -53%. | **PointPillars + Faster R-CNN:** Faster R-CNN +70% to +80% across all rain levels. **PointPillars + ViT-UPerNet:** PointPillars +53% to +73%; ViT-UPerNet up to +72%. **CenterPoint + DINO:** CenterPoint +39% to +57%. **3DSSD + DINO:** DINO up to +43%. |
| P25 | **3DSSD + DeepLabV3+:** DeepLabV3+ +11% to +26% across all rain levels. **CenterPoint + YOLOv3:** YOLOv3 +12% to +13% in selected conditions. | No pair exceeded 10%. |
| P50 | **CenterPoint + DINO:** CenterPoint -13% to -14% across all rain levels. | **PointPillars + ViT-UPerNet:** PointPillars -11% to -14% across all rain levels. **CenterPoint + DETR:** DETR -14.7% at 50 mm/h. **CenterPoint + YOLOv3:** YOLOv3 +11.5% at 25 mm/h. |
| P99 | **PointPillars + YOLOv3:** YOLOv3 -11.2% at 7.5 mm/h. | No pair exceeded 10%. |

## Pair-level takeaways

- **CenterPoint + DINO, non-MPS:** strongest center-to-tail reshaping.
  CenterPoint's median decreases while normalized upper-tail separation grows
  by roughly two to three times.
- **PointPillars + Faster R-CNN, MPS:** strongest persistent tail-ratio
  increase, approximately 70% to 80%.
- **PointPillars + ViT-UPerNet, MPS:** combines a persistent PointPillars
  median decrease with large relative-tail growth.
- **3DSSD + Mask R-CNN, MPS:** strongest persistent CV increase.
- **3DSSD + DeepLabV3+, non-MPS:** strongest persistent P25 shift.
- **3DSSD + DINO, MPS:** DINO has strong range and tail variation while its
  central quantiles remain comparatively stable.

The 795% PointPillars CV increase with YOLOv3 is a sharp one-condition
dispersion spike rather than a matching movement of the central quantiles.

## Research framing

> Model pairing determines which component of the latency distribution
> responds to adverse inputs. The response may appear as a center shift,
> lower-quartile shift, or expansion and contraction of relative tail
> variability. This pair-conditioned response also changes with GPU sharing
> mode.

Scheduling and model-selection policies should therefore use condition-aware,
pair-specific distribution profiles rather than only isolated-model averages.

## Scope restrictions

These results describe one execution per condition. CV includes every frame
and is especially sensitive to extreme observations, while the normalized
tail ratio can change because P50, P99, or both move. The measurements support
distribution-shape comparisons for the tested pairs and modes; statistical
and causal claims require repeated executions.

## Data sources

- `../frame_metrics.csv`
- `../weather_comparisons_mps_off.csv`
- `../weather_comparisons_mps_on.csv`
- `../non_mps_summary.csv`
- `../mps_summary.csv`
