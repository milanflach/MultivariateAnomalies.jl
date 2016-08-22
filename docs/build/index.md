
<a id='MultivariateAnomalies.jl-1'></a>

# MultivariateAnomalies.jl


*Decting multivariate anomalies in Julia.*


A package for detecting anomalies in multivariate data.


*Keywords: Novelty detection, Anomaly Detection, Outlier Detection, Statistical Process Control*


```
Please cite this package as ...
```


<a id='Requirements-1'></a>

## Requirements


  * Julia `0.4`
  * `Distances`, `LIBSVM`, `MultivariateStats`


<a id='Package-Features-1'></a>

## Package Features


  * Extract the relevant features from the data  ```@contents Pages = ["man/FeatureExtraction.md"] ```
  * Detect the anomalies ```@contents Pages = ["man/DetectionAlgorithms.md"] ```
  * Postprocess your anomaly scores, by computing their quantiles or ensembles ```@contents Pages = ["man/Scores"] ```
  * Compute the area under the curve as external evaluation metric ```@contents Pages = ["man/AUC"] ```


<a id='Using-the-Package-1'></a>

## Using the Package


<a id='Index-1'></a>

## Index

- [`MultivariateAnomalies.EWMA`](man/FeatureExtraction.md#MultivariateAnomalies.EWMA)
- [`MultivariateAnomalies.EWMA!`](man/FeatureExtraction.md#MultivariateAnomalies.EWMA!)
- [`MultivariateAnomalies.TDE`](man/FeatureExtraction.md#MultivariateAnomalies.TDE)
- [`MultivariateAnomalies.get_MedianCycle`](man/FeatureExtraction.md#MultivariateAnomalies.get_MedianCycle)
- [`MultivariateAnomalies.get_MedianCycle!`](man/FeatureExtraction.md#MultivariateAnomalies.get_MedianCycle!)
- [`MultivariateAnomalies.get_MedianCycles`](man/FeatureExtraction.md#MultivariateAnomalies.get_MedianCycles)
- [`MultivariateAnomalies.globalICA`](man/FeatureExtraction.md#MultivariateAnomalies.globalICA)
- [`MultivariateAnomalies.globalPCA`](man/FeatureExtraction.md#MultivariateAnomalies.globalPCA)
- [`MultivariateAnomalies.init_MedianCycle`](man/FeatureExtraction.md#MultivariateAnomalies.init_MedianCycle)
- [`MultivariateAnomalies.mw_COR`](man/FeatureExtraction.md#MultivariateAnomalies.mw_COR)
- [`MultivariateAnomalies.mw_VAR`](man/FeatureExtraction.md#MultivariateAnomalies.mw_VAR)
- [`MultivariateAnomalies.sMSC`](man/FeatureExtraction.md#MultivariateAnomalies.sMSC)
- [`MultivariateAnomalies.KDE`](man/DetectionAlgorithms.md#MultivariateAnomalies.KDE)
- [`MultivariateAnomalies.KDE!`](man/DetectionAlgorithms.md#MultivariateAnomalies.KDE!)
- [`MultivariateAnomalies.KNN_Delta`](man/DetectionAlgorithms.md#MultivariateAnomalies.KNN_Delta)
- [`MultivariateAnomalies.KNN_Delta!`](man/DetectionAlgorithms.md#MultivariateAnomalies.KNN_Delta!)
- [`MultivariateAnomalies.KNN_Gamma`](man/DetectionAlgorithms.md#MultivariateAnomalies.KNN_Gamma)
- [`MultivariateAnomalies.KNN_Gamma!`](man/DetectionAlgorithms.md#MultivariateAnomalies.KNN_Gamma!)
- [`MultivariateAnomalies.REC`](man/DetectionAlgorithms.md#MultivariateAnomalies.REC)
- [`MultivariateAnomalies.REC!`](man/DetectionAlgorithms.md#MultivariateAnomalies.REC!)
- [`MultivariateAnomalies.SVDD_predict`](man/DetectionAlgorithms.md#MultivariateAnomalies.SVDD_predict)
- [`MultivariateAnomalies.SVDD_predict!`](man/DetectionAlgorithms.md#MultivariateAnomalies.SVDD_predict!)
- [`MultivariateAnomalies.SVDD_train`](man/DetectionAlgorithms.md#MultivariateAnomalies.SVDD_train)
- [`MultivariateAnomalies.T2`](man/DetectionAlgorithms.md#MultivariateAnomalies.T2)
- [`MultivariateAnomalies.T2!`](man/DetectionAlgorithms.md#MultivariateAnomalies.T2!)
- [`MultivariateAnomalies.UNIV`](man/DetectionAlgorithms.md#MultivariateAnomalies.UNIV)
- [`MultivariateAnomalies.UNIV!`](man/DetectionAlgorithms.md#MultivariateAnomalies.UNIV!)
- [`MultivariateAnomalies.init_KDE`](man/DetectionAlgorithms.md#MultivariateAnomalies.init_KDE)
- [`MultivariateAnomalies.init_KNN_Delta`](man/DetectionAlgorithms.md#MultivariateAnomalies.init_KNN_Delta)
- [`MultivariateAnomalies.init_KNN_Gamma`](man/DetectionAlgorithms.md#MultivariateAnomalies.init_KNN_Gamma)
- [`MultivariateAnomalies.init_REC`](man/DetectionAlgorithms.md#MultivariateAnomalies.init_REC)
- [`MultivariateAnomalies.init_SVDD_predict`](man/DetectionAlgorithms.md#MultivariateAnomalies.init_SVDD_predict)
- [`MultivariateAnomalies.init_T2`](man/DetectionAlgorithms.md#MultivariateAnomalies.init_T2)
- [`MultivariateAnomalies.init_UNIV`](man/DetectionAlgorithms.md#MultivariateAnomalies.init_UNIV)

