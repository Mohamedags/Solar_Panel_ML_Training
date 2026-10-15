# Experimental Results

This directory contains the validation metrics, training plots, and class-weight comparison for the solar-panel defect detection experiments.

## 1. Experiment Summary

The experiments investigate two hotspot-classification approaches:

- **Method A:** combines all hotspots into one class.
- **Method B:** separates hotspots into `hotspot_obvious` and `hotspot_doubt`, with different classification weights tested for the doubtful-hotspot class.

The main results are summarized in `experiment_summary.csv`. Per-class results are provided in `metrics/per_class_metrics.csv`.

## 2. Overall Validation Results

### Method A

| Metric | Value |
|---|---:|
| Precision | 0.882 |
| Recall | 0.831 |
| mAP@0.50 | 0.876 |
| mAP@0.50:0.95 | 0.614 |

### Method B: Class-Weight Comparison

| Doubtful-hotspot weight | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
|---:|---:|---:|---:|---:|
| 1.0 | 0.769 | 0.781 | 0.806 | 0.559 |
| 0.7 | 0.774 | 0.804 | 0.825 | 0.560 |
| 0.4 | 0.791 | 0.804 | 0.819 | 0.550 |

Among the tested Method B configurations, a weight of 0.7 achieved the highest mAP@0.50 and mAP@0.50:0.95. A weight of 0.4 achieved the highest precision.

These results are descriptive. Each configuration was evaluated in a single training run, and further experiments would be needed to assess the stability of these differences.

The overall metrics for Methods A and B are not directly comparable because their hotspot class definitions differ. A fair comparison requires evaluation under a common class taxonomy and consistent evaluation settings.

## 3. Training and Validation Plots

The `plots/` directory contains selected figures generated during training and validation.

- `results.png`: training and validation loss and performance metrics across epochs.
- `BoxPR_curve.png`: precision–recall trade-off across confidence thresholds.
- `BoxF1_curve.png`: F1 score across confidence thresholds.
- `confusion_matrix_normalized.png`: normalized class-confusion patterns.

Each plot is organized by method and experiment. Refer to the corresponding experiment identifier when comparing figures.

## 4. Class-Weight Comparison

The `weight_comparison/` directory contains:

- `weight_comparison.csv`: numerical validation metrics for each tested class weight.
- `weight_comparison.png`: visual comparison of precision, recall, mAP@0.50, and mAP@0.50:0.95.

Changing the doubtful-hotspot class weight affects the balance between precision, recall, and mAP. The best setting depends on the intended application and the relative importance of missed detections and false positives.

## 5. CSV File Descriptions

### `experiment_summary.csv`

Contains one row per training run, including the experiment identifier, method, training configuration, validation metrics, training duration, and notes.

### `metrics/per_class_metrics.csv`

Contains one row per class per experiment, allowing performance to be examined separately for panels, hotspots, and scratches.

### `weight_comparison/weight_comparison.csv`

Contains the overall validation metrics for each tested `hotspot_doubt` weight in Method B.

## 6. Limitations

- The validation set is limited in size.
- The obvious-hotspot class has relatively few validation instances.
- Each class-weight setting was evaluated in a single run.
- Methods A and B use different class definitions, limiting direct comparison of their aggregate metrics.
- Results depend on the dataset split, annotations, training configuration, and evaluation settings.

Prediction examples may be added after permission to publish them has been confirmed.