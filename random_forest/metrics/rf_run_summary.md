# Random-Forest experiment summary

Parsed *6* runs located in `metrics/`.

## Comparison table

| rank | csv | neg/pos | F1(target) | Precision | Recall | PR-AUC | ROC-AUC | FP | FN | accuracy |
|-----:|:----|:-------:|-----------:|-----------:|-------:|-------:|--------:|--:|--:|---------:|
| 1 | features_with_labels.csv | 2 | 0.885 | 0.861 | 0.910 | 0.919 | 0.966 | 23 | 14 | 0.911 |
| 2 | mfcc_with_labels.csv | 1 | 0.879 | 0.800 | 0.974 | 0.924 | 0.963 | 38 | 4 | 0.900 |
| 3 | features_with_labels.csv | 1 | 0.876 | 0.796 | 0.974 | 0.923 | 0.964 | 39 | 4 | 0.897 |
| 4 | mfcc_with_labels.csv | 2 | 0.868 | 0.852 | 0.885 | 0.913 | 0.960 | 24 | 18 | 0.900 |
| 5 | spectral_with_labels.csv | 1 | 0.813 | 0.832 | 0.795 | 0.892 | 0.928 | 25 | 32 | 0.864 |
| 6 | spectral_with_labels.csv | 2 | 0.769 | 0.920 | 0.660 | 0.888 | 0.915 | 9 | 53 | 0.852 |

## Overall observations


* **Best F1** : 0.885;  **median F1** : 0.872
* **Best PR-AUC** : 0.924;  **median PR-AUC** : 0.916

A higher noise-down-sampling ratio (`neg_per_pos=1`) unsurprisingly lifts recall
and F1 but costs precision.  The MFCC+Spectral feature set (`features_with_labels.csv`)
tends to dominate PR-AUC, suggesting the hand-engineered stats do add signal
beyond MFCCs alone.

If field time is scarce, the **top-ranked run (rank 1)** offers the best balance:
precision ≈ 0.86 with recall ≈ 0.91.
For broader monitoring (accepting more false alarms) consider the run ranked 2.
