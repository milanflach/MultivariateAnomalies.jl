
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
  * `Distances`, `MultivariateStats`
  *  latest `LIBSVM` branch via: `Pkg.clone("https://github.com/milanflach/LIBSVM.jl.git")` `Pkg.checkout("LIBSVM", "mutating_versions")` `Pkg.build("LIBSVM")`


<a id='Package-Features-1'></a>

## Package Features


  * Extract the relevant features from the data  ```@contents Pages = ["man/FeatureExtraction.md"] ```
  * Compute Distance, Kernel matrices and k-nearest neighbors objects  ```@contents Pages = ["man/DistDensity.md"] ```
  * Detect the anomalies ```@contents Pages = ["man/DetectionAlgorithms.md"] ```
  * Postprocess your anomaly scores, by computing their quantiles or ensembles ```@contents Pages = ["man/Scores"] ```
  * Compute the area under the curve as external evaluation metric ```@contents Pages = ["man/AUC"] ```


<a id='Using-the-Package-1'></a>

## Using the Package


We provide high-level convenience functions for detecting the anomalies. Namely the pair of 


`P = getParameters(algorithms, training_data)` and `detectAnomalies(testing_data, P)`


sets standard choices of the Parameters `P` and hands the parameters as well as the algorithms choice over to detect the anomalies. 


Currently supported algorithms include Kernel Density Estimation (`algorithms = ["KDE"]`), Recurrences (`"REC"`), k-Nearest Neighbors algorithms (`"KNN-Gamma"`, `"KNN-Delta"`), Hotelling's T^2 (`"T2"`), Support Vector Data Description (`"SVDD"`) and Kernel Null Foley Summon Transform (`"KNFST"`). With `getParameters()` it is also possible to compute output scores of multiple algorithms at once (`algorihtms = ["KDE", "T2"]`), quantiles of the output anomaly scores (`quantiles = true`) and ensembles of the selected algorithms (e.g. `ensemble_method = "mean"`). For more details about the detection algorithms and their usage please consider 


```
@contents Pages = ["man/DetectionAlgorithms.md"]
```


<a id='Input-Data-1'></a>

## Input Data


Within MultivariateAnomalies we assume that observations/samples/time steps are stored along the first dimension of the data array (rows of a matrix) with the number of observations `T = size(data, 1)`. Variables/attributes are stored along the last dimension `N` of the data array (along the columns of a matrix) with the number of variables `VAR = size(data, N)`. We are interested in the question which observation(s) of the data are anomalous.


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
- [`MultivariateAnomalies.KNFST_predict`](man/DetectionAlgorithms.md#MultivariateAnomalies.KNFST_predict)
- [`MultivariateAnomalies.KNFST_predict!`](man/DetectionAlgorithms.md#MultivariateAnomalies.KNFST_predict!)
- [`MultivariateAnomalies.KNFST_train`](man/DetectionAlgorithms.md#MultivariateAnomalies.KNFST_train)
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
- [`MultivariateAnomalies.detectAnomalies`](man/DetectionAlgorithms.md#MultivariateAnomalies.detectAnomalies)
- [`MultivariateAnomalies.detectAnomalies!`](man/DetectionAlgorithms.md#MultivariateAnomalies.detectAnomalies!)
- [`MultivariateAnomalies.getParameters`](man/DetectionAlgorithms.md#MultivariateAnomalies.getParameters)
- [`MultivariateAnomalies.init_KDE`](man/DetectionAlgorithms.md#MultivariateAnomalies.init_KDE)
- [`MultivariateAnomalies.init_KNFST`](man/DetectionAlgorithms.md#MultivariateAnomalies.init_KNFST)
- [`MultivariateAnomalies.init_KNN_Delta`](man/DetectionAlgorithms.md#MultivariateAnomalies.init_KNN_Delta)
- [`MultivariateAnomalies.init_KNN_Gamma`](man/DetectionAlgorithms.md#MultivariateAnomalies.init_KNN_Gamma)
- [`MultivariateAnomalies.init_REC`](man/DetectionAlgorithms.md#MultivariateAnomalies.init_REC)
- [`MultivariateAnomalies.init_SVDD_predict`](man/DetectionAlgorithms.md#MultivariateAnomalies.init_SVDD_predict)
- [`MultivariateAnomalies.init_T2`](man/DetectionAlgorithms.md#MultivariateAnomalies.init_T2)
- [`MultivariateAnomalies.init_UNIV`](man/DetectionAlgorithms.md#MultivariateAnomalies.init_UNIV)
- [`MultivariateAnomalies.init_detectAnomalies`](man/DetectionAlgorithms.md#MultivariateAnomalies.init_detectAnomalies)

