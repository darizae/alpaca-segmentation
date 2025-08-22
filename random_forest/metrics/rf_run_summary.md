# Random-Forest experiment summary

Parsed *7* runs located in `metrics/`.

## Comparison table

| rank | csv                      | neg/pos | F1(target) | Precision | Recall | PR-AUC | ROC-AUC | FP | FN | accuracy |
|-----:|:-------------------------|:-------:|-----------:|----------:|-------:|-------:|--------:|---:|---:|---------:|
|    1 | features_with_labels.csv |    2    |      0.885 |     0.861 |  0.910 |  0.919 |   0.966 | 23 | 14 |    0.911 |
|    2 | features_with_labels.csv |    1    |      0.876 |     0.796 |  0.974 |  0.923 |   0.964 | 39 |  4 |    0.897 |
|    3 | actually_with_logits.csv |    1    |      0.876 |     0.815 |  0.946 |  0.910 |   0.966 | 32 |  8 |    0.914 |
|    4 | actually_with_logits.csv |    2    |      0.875 |     0.884 |  0.866 |  0.917 |   0.969 | 17 | 20 |    0.921 |
|    5 | features_py.csv          |    1    |      0.858 |     0.783 |  0.949 |  0.932 |   0.959 | 41 |  8 |    0.883 |
|    6 | actually_with_logits.csv |    3    |      0.848 |     0.896 |  0.805 |  0.919 |   0.969 | 14 | 29 |    0.908 |
|    7 | features_py.csv          |    2    |      0.840 |     0.898 |  0.788 |  0.934 |   0.959 | 14 | 33 |    0.888 |

## Overall observations


* **Best F1** : 0.885;  **median F1** : 0.875
* **Best PR-AUC** : 0.934;  **median PR-AUC** : 0.919

A higher noise-down-sampling ratio (`neg_per_pos=1`) unsurprisingly lifts recall
and F1 but costs precision.  The MFCC+Spectral feature set (`features_with_labels.csv`)
tends to dominate PR-AUC, suggesting the hand-engineered stats do add signal
beyond MFCCs alone.

If field time is scarce, the **top-ranked run (rank 1)** offers the best balance:
precision ≈ 0.86 with recall ≈ 0.91.
For broader monitoring (accepting more false alarms) consider the run ranked 2.
