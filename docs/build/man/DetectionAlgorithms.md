
<a id='Detection-Algorithms-1'></a>

## Detection Algorithms


detect anomalies out of multivariate correlated data.


<a id='Functions-1'></a>

## Functions

<a id='MultivariateAnomalies.REC' href='#MultivariateAnomalies.REC'>#</a>
**`MultivariateAnomalies.REC`** &mdash; *Function*.



```
REC(D::AbstractArray, rec_threshold::Float64, temp_excl::Int = 5)
```

Count the number of observations (recurrences) which fall into a radius `rec_threshold` of a distance matrix `D`. Exclude steps which are closer than `temp_excl` to be count as recurrences (default: `temp_excl = 5`)

<a id='MultivariateAnomalies.REC!' href='#MultivariateAnomalies.REC!'>#</a>
**`MultivariateAnomalies.REC!`** &mdash; *Function*.



```
REC!(rec_out::AbstractArray, D::AbstractArray, rec_threshold::Float64, temp_excl::Int = 5)
```

Memory efficient version of `REC()` for use within a loop. rec_out is preallocated output, should be initialised with `init_REC()`.

<a id='MultivariateAnomalies.init_REC' href='#MultivariateAnomalies.init_REC'>#</a>
**`MultivariateAnomalies.init_REC`** &mdash; *Function*.



```
init_REC(D::Array{Float64, 2})
init_REC(T::Int)
```

get object for memory efficient `REC!()` versions. Input can be a distance matrix `D` or the number of timesteps (observations) `T`.

<a id='MultivariateAnomalies.KDE' href='#MultivariateAnomalies.KDE'>#</a>
**`MultivariateAnomalies.KDE`** &mdash; *Function*.



```
KDE(K)
```

Compute a Kernel Density Estimation (the Parzen sum), given a Kernel matrix `K`.

<a id='MultivariateAnomalies.KDE!' href='#MultivariateAnomalies.KDE!'>#</a>
**`MultivariateAnomalies.KDE!`** &mdash; *Function*.



```
KDE!(KDE_out, K)
```

Memory efficient version of `KDE()`. Additionally uses preallocated `KDE_out` object for writing the results. Initialize `KDE_out` with `init_KDE()`.

<a id='MultivariateAnomalies.init_KDE' href='#MultivariateAnomalies.init_KDE'>#</a>
**`MultivariateAnomalies.init_KDE`** &mdash; *Function*.



```
init_KDE(K::Array{Float64, 2})
init_KDE(T::Int)
```

Returns `KDE_out` object for usage in `KDE!()`. Use either a Kernel matrix `K` or the number of timesteps `T` as argument.

<a id='MultivariateAnomalies.T2' href='#MultivariateAnomalies.T2'>#</a>
**`MultivariateAnomalies.T2`** &mdash; *Function*.



```
T2{tp}(data::AbstractArray{tp,2}, Q::AbstractArray[, mv])
```

Compute Hotelling's T^2 control chart (the squared Mahalanobis distance to the data's mean vector (`mv`), given the covariance matrix `Q`). Input data is a two dimensional data matrix (time * variables).

<a id='MultivariateAnomalies.T2!' href='#MultivariateAnomalies.T2!'>#</a>
**`MultivariateAnomalies.T2!`** &mdash; *Function*.



```
T2!(t2_out, data, Q[, mv])
```

Memory efficient version of `T2()`, for usage within a loop etc. Initialize the `t2_out` object with `init_T2()`. `t2_out[1]` contains the squred Mahalanobis distance after computation.

<a id='MultivariateAnomalies.init_T2' href='#MultivariateAnomalies.init_T2'>#</a>
**`MultivariateAnomalies.init_T2`** &mdash; *Function*.



```
init_T2(VAR::Int, T::Int)
init_T2{tp}(data::AbstractArray{tp,2})
```

initialize `t2_out` object for `T2!` either with number of varaiables `VAR` and timesteps `T` or with a two dimensional `data` matrix (time * variables)

<a id='MultivariateAnomalies.KNN_Gamma' href='#MultivariateAnomalies.KNN_Gamma'>#</a>
**`MultivariateAnomalies.KNN_Gamma`** &mdash; *Function*.



```
KNN_Gamma(knn_dists_out)
```

This function computes the mean distance of the K nearest neighbors given a `knn_dists_out` object from `knn_dists()` as input argument.

<a id='MultivariateAnomalies.KNN_Gamma!' href='#MultivariateAnomalies.KNN_Gamma!'>#</a>
**`MultivariateAnomalies.KNN_Gamma!`** &mdash; *Function*.



```
KNN_Gamma!(KNN_Gamma_out, knn_dists_out)
```

Memory efficient version of `KNN_Gamma`, to be used in a loop. Initialize `KNN_Gamma_out` with `init_KNN_Gamma()`.

<a id='MultivariateAnomalies.init_KNN_Gamma' href='#MultivariateAnomalies.init_KNN_Gamma'>#</a>
**`MultivariateAnomalies.init_KNN_Gamma`** &mdash; *Function*.



```
init_KNN_Gamma(T::Int)
init_KNN_Gamma(knn_dists_out)
```

initialize a `KNN_Gamma_out` object for `KNN_Gamma!` either with `T`, the number of timesteps or with a `knn_dists_out` object.

<a id='MultivariateAnomalies.KNN_Delta' href='#MultivariateAnomalies.KNN_Delta'>#</a>
**`MultivariateAnomalies.KNN_Delta`** &mdash; *Function*.



```
KNN_Delta(knn_dists_out, data)
```

Compute Delta as vector difference of the K nearest neighbors. Arguments are a `knn_dists()` object (`knn_dists_out`) and a `data` matrix (time * variables)

<a id='MultivariateAnomalies.KNN_Delta!' href='#MultivariateAnomalies.KNN_Delta!'>#</a>
**`MultivariateAnomalies.KNN_Delta!`** &mdash; *Function*.



```
KNN_Delta!(KNN_Delta_out, knn_dists_out, data)
```

Memory Efficient Version of `KNN_Delta()`. `KNN_Delta_out[1]` is the vector difference of the K nearest neighbors.

<a id='MultivariateAnomalies.init_KNN_Delta' href='#MultivariateAnomalies.init_KNN_Delta'>#</a>
**`MultivariateAnomalies.init_KNN_Delta`** &mdash; *Function*.



```
init_KNN_Delta(T, VAR, K)
```

return a `KNN_Delta_out` object to be used for `KNN_Delta!`. Input: timesteps `T`, variables `V`, number of K nearest neighbors `K`.

<a id='MultivariateAnomalies.UNIV' href='#MultivariateAnomalies.UNIV'>#</a>
**`MultivariateAnomalies.UNIV`** &mdash; *Function*.



```
UNIV(data)
```

order the values in each varaible and return their maximum, i.e. any of the variables in `data` (times * variables) is above a given quantile, the highest quantile will be returned.

<a id='MultivariateAnomalies.UNIV!' href='#MultivariateAnomalies.UNIV!'>#</a>
**`MultivariateAnomalies.UNIV!`** &mdash; *Function*.



```
UNIV!(univ_out, data)
```

Memory efficient version of `UNIV()`, input an `univ_out` object from `init_UNIV()` and some `data` matrix time * variables

<a id='MultivariateAnomalies.init_UNIV' href='#MultivariateAnomalies.init_UNIV'>#</a>
**`MultivariateAnomalies.init_UNIV`** &mdash; *Function*.



```
init_UNIV(T::Int, VAR::Int)
init_UNIV{tp}(data::AbstractArray{tp, 2})
```

initialize a `univ_out` object to be used in `UNIV!()` either with number of time steps `T` and variables `V` or with a `data` matrix time * variables.

<a id='MultivariateAnomalies.SVDD_train' href='#MultivariateAnomalies.SVDD_train'>#</a>
**`MultivariateAnomalies.SVDD_train`** &mdash; *Function*.



```
SVDD_train(K, nu)
```

train a one class support vecort machine model (i.e. support vector data description), given a kernel matrix K and and the highest possible percentage of outliers `nu`. Returns the model object (`svdd_model`). Requires LIBSVM.

<a id='MultivariateAnomalies.SVDD_predict' href='#MultivariateAnomalies.SVDD_predict'>#</a>
**`MultivariateAnomalies.SVDD_predict`** &mdash; *Function*.



```
SVDD_predict(K, svdd_model)
```

predict the outlierness of an object given the testing Kernel matrix `K` and the `svdd_model` from SVDD_train(). Requires LIBSVM.

<a id='MultivariateAnomalies.SVDD_predict!' href='#MultivariateAnomalies.SVDD_predict!'>#</a>
**`MultivariateAnomalies.SVDD_predict!`** &mdash; *Function*.



```
SVDD_predict!(SVDD_out, svdd_model, K)
```

Memory efficient version of `SVDD_predict()`. Additional input argument is the `SVDD_out` object from `init_SVDD_predict()`. `SVDD_out[1]` are predicted labels, `SVDD_out[2]` decision_values. Requires LIBSVM.

<a id='MultivariateAnomalies.init_SVDD_predict' href='#MultivariateAnomalies.init_SVDD_predict'>#</a>
**`MultivariateAnomalies.init_SVDD_predict`** &mdash; *Function*.



```
init_SVDD_predict(T::Int)
init_SVDD_predict(T::Int, Ttrain::Int)
```

initializes a `SVDD_out` object to be used in `SVDD_predict!()`. Input is the number of time steps `T` (in prediction mode). If `T` for prediction differs from T of the training data (`Ttrain`) use `Ttrain` as additional argument.

<a id='MultivariateAnomalies.KNFST_train' href='#MultivariateAnomalies.KNFST_train'>#</a>
**`MultivariateAnomalies.KNFST_train`** &mdash; *Function*.



```
KNFST_train(K)
```

train a one class novelty KNFST model on a Kernel matrix `K` according to Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler: "Kernel Null Space Methods for Novelty Detection". Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013.

**Output**

`(proj, targetValue)` `proj` 	– projection vector for data points (project x via kx*proj, where kx is row vector containing kernel values of x and training data) `targetValue` – value of all training samples in the null space

<a id='MultivariateAnomalies.KNFST_predict' href='#MultivariateAnomalies.KNFST_predict'>#</a>
**`MultivariateAnomalies.KNFST_predict`** &mdash; *Function*.



```
KNFST_predict(model, K)
```

predict the outlierness of some data (represented by the kernel matrix `K`), given some KNFST `model` from `KNFST_train(K)`.

Please cite: Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler: "Kernel Null Space Methods for Novelty Detection". Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013.

<a id='MultivariateAnomalies.KNFST_predict!' href='#MultivariateAnomalies.KNFST_predict!'>#</a>
**`MultivariateAnomalies.KNFST_predict!`** &mdash; *Function*.



```
KNFST_predict!(KNFST_out, KNFST_mod, K)
```

predict the outlierness of some data (represented by the kernel matrix `K`), given a `KNFST_out` object (`init_KNFST()`), some KNFST model (`KNFST_mod = KNFST_train(K)`) and the testing kernel matrix K.

Please cite: Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler: "Kernel Null Space Methods for Novelty Detection". Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013.

<a id='MultivariateAnomalies.init_KNFST' href='#MultivariateAnomalies.init_KNFST'>#</a>
**`MultivariateAnomalies.init_KNFST`** &mdash; *Function*.



```
init_KNFST(T, KNFST_mod)
```

initialize a `KNFST_out`object for the use with `KNFST_predict!`, given `T`, the number of observations and the model output `KNFST_train(K)`.


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

