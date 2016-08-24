
<a id='Scores-1'></a>

## Scores


Postprocess your anomaly scores by making different algorithms comparable and computing their ensemble.


<a id='Functions-1'></a>

## Functions

<a id='MultivariateAnomalies.get_quantile_scores' href='#MultivariateAnomalies.get_quantile_scores'>#</a>
**`MultivariateAnomalies.get_quantile_scores`** &mdash; *Function*.



```
get_quantile_scores(scores, quantiles = 0.0:0.01:1.0)
```

return the quantiles of the given N dimensional anomaly `scores` cube. `quantiles` (default: `quantiles = 0.0:0.01:1.0`) is a Float range of quantiles. Any score being greater or equal `quantiles[i]` and beeing smaller than `quantiles[i+1]` is assigned to the respective quantile `quantiles[i]`.

**Examples**

```jlcon
julia> scores1 = rand(10, 2)
julia> quantile_scores1 = get_quantile_scores(scores1)
```

<a id='MultivariateAnomalies.compute_ensemble' href='#MultivariateAnomalies.compute_ensemble'>#</a>
**`MultivariateAnomalies.compute_ensemble`** &mdash; *Function*.



```
compute_ensemble(m1_scores, m2_scores[, m3_scores, m4_scores], ensemble = "mean")
```

compute the mean (`ensemble = "mean"`), minimum (`ensemble = "min"`), maximum (`ensemble = "max"`) or median (`ensemble = "median"`) of the given anomaly scores. Supports between 2 and 4 scores input arrays (`m1_scores, ..., m4_scores`). The scores of the different anomaly detection algorithms should be somehow comparable, e.g., by using `get_quantile_scores()` before.

**Examples**

```jlcon
julia> scores1 = rand(10, 2)
julia> scores2 = rand(10, 2)
julia> quantile_scores1 = get_quantile_scores(scores1)
julia> quantile_scores2 = get_quantile_scores(scores2)
julia> compute_ensemble(quantile_scores1, quantile_scores2, ensemble = "max")
```


<a id='Index-1'></a>

## Index

- [`MultivariateAnomalies.EWMA`](FeatureExtraction.md#MultivariateAnomalies.EWMA)
- [`MultivariateAnomalies.EWMA!`](FeatureExtraction.md#MultivariateAnomalies.EWMA!)
- [`MultivariateAnomalies.KDE`](DetectionAlgorithms.md#MultivariateAnomalies.KDE)
- [`MultivariateAnomalies.KDE!`](DetectionAlgorithms.md#MultivariateAnomalies.KDE!)
- [`MultivariateAnomalies.KNFST_predict`](DetectionAlgorithms.md#MultivariateAnomalies.KNFST_predict)
- [`MultivariateAnomalies.KNFST_predict!`](DetectionAlgorithms.md#MultivariateAnomalies.KNFST_predict!)
- [`MultivariateAnomalies.KNFST_train`](DetectionAlgorithms.md#MultivariateAnomalies.KNFST_train)
- [`MultivariateAnomalies.KNN_Delta`](DetectionAlgorithms.md#MultivariateAnomalies.KNN_Delta)
- [`MultivariateAnomalies.KNN_Delta!`](DetectionAlgorithms.md#MultivariateAnomalies.KNN_Delta!)
- [`MultivariateAnomalies.KNN_Gamma`](DetectionAlgorithms.md#MultivariateAnomalies.KNN_Gamma)
- [`MultivariateAnomalies.KNN_Gamma!`](DetectionAlgorithms.md#MultivariateAnomalies.KNN_Gamma!)
- [`MultivariateAnomalies.REC`](DetectionAlgorithms.md#MultivariateAnomalies.REC)
- [`MultivariateAnomalies.REC!`](DetectionAlgorithms.md#MultivariateAnomalies.REC!)
- [`MultivariateAnomalies.SVDD_predict`](DetectionAlgorithms.md#MultivariateAnomalies.SVDD_predict)
- [`MultivariateAnomalies.SVDD_predict!`](DetectionAlgorithms.md#MultivariateAnomalies.SVDD_predict!)
- [`MultivariateAnomalies.SVDD_train`](DetectionAlgorithms.md#MultivariateAnomalies.SVDD_train)
- [`MultivariateAnomalies.T2`](DetectionAlgorithms.md#MultivariateAnomalies.T2)
- [`MultivariateAnomalies.T2!`](DetectionAlgorithms.md#MultivariateAnomalies.T2!)
- [`MultivariateAnomalies.TDE`](FeatureExtraction.md#MultivariateAnomalies.TDE)
- [`MultivariateAnomalies.UNIV`](DetectionAlgorithms.md#MultivariateAnomalies.UNIV)
- [`MultivariateAnomalies.UNIV!`](DetectionAlgorithms.md#MultivariateAnomalies.UNIV!)
- [`MultivariateAnomalies.auc`](AUC.md#MultivariateAnomalies.auc)
- [`MultivariateAnomalies.auc_fpr_tpr`](AUC.md#MultivariateAnomalies.auc_fpr_tpr)
- [`MultivariateAnomalies.boolevents`](AUC.md#MultivariateAnomalies.boolevents)
- [`MultivariateAnomalies.compute_ensemble`](Scores.md#MultivariateAnomalies.compute_ensemble)
- [`MultivariateAnomalies.get_MedianCycle`](FeatureExtraction.md#MultivariateAnomalies.get_MedianCycle)
- [`MultivariateAnomalies.get_MedianCycle!`](FeatureExtraction.md#MultivariateAnomalies.get_MedianCycle!)
- [`MultivariateAnomalies.get_MedianCycles`](FeatureExtraction.md#MultivariateAnomalies.get_MedianCycles)
- [`MultivariateAnomalies.get_quantile_scores`](Scores.md#MultivariateAnomalies.get_quantile_scores)
- [`MultivariateAnomalies.globalICA`](FeatureExtraction.md#MultivariateAnomalies.globalICA)
- [`MultivariateAnomalies.globalPCA`](FeatureExtraction.md#MultivariateAnomalies.globalPCA)
- [`MultivariateAnomalies.init_KDE`](DetectionAlgorithms.md#MultivariateAnomalies.init_KDE)
- [`MultivariateAnomalies.init_KNFST`](DetectionAlgorithms.md#MultivariateAnomalies.init_KNFST)
- [`MultivariateAnomalies.init_KNN_Delta`](DetectionAlgorithms.md#MultivariateAnomalies.init_KNN_Delta)
- [`MultivariateAnomalies.init_KNN_Gamma`](DetectionAlgorithms.md#MultivariateAnomalies.init_KNN_Gamma)
- [`MultivariateAnomalies.init_MedianCycle`](FeatureExtraction.md#MultivariateAnomalies.init_MedianCycle)
- [`MultivariateAnomalies.init_REC`](DetectionAlgorithms.md#MultivariateAnomalies.init_REC)
- [`MultivariateAnomalies.init_SVDD_predict`](DetectionAlgorithms.md#MultivariateAnomalies.init_SVDD_predict)
- [`MultivariateAnomalies.init_T2`](DetectionAlgorithms.md#MultivariateAnomalies.init_T2)
- [`MultivariateAnomalies.init_UNIV`](DetectionAlgorithms.md#MultivariateAnomalies.init_UNIV)
- [`MultivariateAnomalies.mw_COR`](FeatureExtraction.md#MultivariateAnomalies.mw_COR)
- [`MultivariateAnomalies.mw_VAR`](FeatureExtraction.md#MultivariateAnomalies.mw_VAR)
- [`MultivariateAnomalies.sMSC`](FeatureExtraction.md#MultivariateAnomalies.sMSC)

