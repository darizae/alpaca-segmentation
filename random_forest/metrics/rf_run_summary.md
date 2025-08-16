# Random-Forest experiment summary

Parsed *12* runs located in `metrics/`.

## Comparison table

| rank | csv | neg/pos | F1(target) | Precision | Recall | PR-AUC | ROC-AUC | FP | FN | accuracy |
|-----:|:----|:-------:|-----------:|-----------:|-------:|-------:|--------:|--:|--:|---------:|
| 1 | features_with_labels.csv | 2 | 0.885 | 0.861 | 0.910 | 0.919 | 0.966 | 23 | 14 | 0.911 |
| 2 | mfcc_with_labels.csv | 1 | 0.879 | 0.800 | 0.974 | 0.924 | 0.963 | 38 | 4 | 0.900 |
| 3 | features_with_labels.csv | 1 | 0.876 | 0.796 | 0.974 | 0.923 | 0.964 | 39 | 4 | 0.897 |
| 4 | mfcc_with_labels.csv | 2 | 0.868 | 0.852 | 0.885 | 0.913 | 0.960 | 24 | 18 | 0.900 |
| 5 | features_py.csv | 1 | 0.858 | 0.783 | 0.949 | 0.932 | 0.959 | 41 | 8 | 0.883 |
| 6 | features_py.csv | 2 | 0.840 | 0.898 | 0.788 | 0.934 | 0.959 | 14 | 33 | 0.888 |
| 7 | mfcc_py.csv | 2 | 0.835 | 0.879 | 0.795 | 0.921 | 0.948 | 17 | 32 | 0.883 |
| 8 | mfcc_py.csv | 1 | 0.818 | 0.718 | 0.949 | 0.918 | 0.949 | 58 | 8 | 0.842 |
| 9 | spectral_with_labels.csv | 1 | 0.813 | 0.832 | 0.795 | 0.892 | 0.928 | 25 | 32 | 0.864 |
| 10 | spectral_with_labels.csv | 2 | 0.769 | 0.920 | 0.660 | 0.888 | 0.915 | 9 | 53 | 0.852 |
| 11 | spectral_robust_py.csv | 1 | 0.755 | 0.781 | 0.731 | 0.854 | 0.910 | 32 | 42 | 0.823 |
| 12 | spectral_robust_py.csv | 2 | 0.716 | 0.857 | 0.615 | 0.845 | 0.900 | 16 | 60 | 0.818 |

## Overall observations


* **Best F1** : 0.885;  **median F1** : 0.837
* **Best PR-AUC** : 0.934;  **median PR-AUC** : 0.919

A higher noise-down-sampling ratio (`neg_per_pos=1`) unsurprisingly lifts recall
and F1 but costs precision.  The MFCC+Spectral feature set (`features_with_labels.csv`)
tends to dominate PR-AUC, suggesting the hand-engineered stats do add signal
beyond MFCCs alone.

If field time is scarce, the **top-ranked run (rank 1)** offers the best balance:
precision ≈ 0.86 with recall ≈ 0.91.
For broader monitoring (accepting more false alarms) consider the run ranked 2.
