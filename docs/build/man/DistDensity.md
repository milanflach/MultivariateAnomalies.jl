
<a id='Distance,-Kernel-Matrices-and-k-Nearest-Neighbours-1'></a>

## Distance, Kernel Matrices and k-Nearest Neighbours


Compute distance matrices (similarity matrices), convert them into kernel matrices or k-Neartest Neighbor objects.


<a id='Functions-1'></a>

## Functions

<a id='MultivariateAnomalies.dist_matrix' href='#MultivariateAnomalies.dist_matrix'>#</a>
**`MultivariateAnomalies.dist_matrix`** &mdash; *Function*.



```
dist_matrix{tp, N}(data::AbstractArray{tp, N}; dist::ASCIIString = "Euclidean", space::Int = 0, lat::Int = 0, lon::Int = 0, Q = 0)
dist_matrix{tp, N}(data::AbstractArray{tp, N}, training_data; dist::ASCIIString = "Euclidean", space::Int = 0, lat::Int = 0, lon::Int = 0, Q = 0)
```

compute the distance matrix of `data` or the distance matrix between data and training data i.e. the pairwise distances along the first dimension of data, using the last dimension as variables. `dist` is a distance metric, currently `Euclidean`(default), `SqEuclidean`, `Chebyshev`, `Cityblock`, `JSDivergence`, `Mahalanobis` and `SqMahalanobis` are supported. The latter two need a covariance matrix `Q` as input argument.

**Examples**

```jlcon
julia> dc = randn(10, 4,3)
julia> D = dist_matrix(dc, space = 2)
```

<a id='MultivariateAnomalies.dist_matrix!' href='#MultivariateAnomalies.dist_matrix!'>#</a>
**`MultivariateAnomalies.dist_matrix!`** &mdash; *Function*.



```
dist_matrix!
```

compute the distance matrix of `data`, similar to `dist_matrix()`. `D_out` object has to be preallocated, i.e. with `init_dist_matrix`.

```jlcon
julia> dc = randn(10,4, 4,3)
julia> D_out = init_dist_matrix(dc)
julia> dist_matrix!(D_out, dc, lat = 2, lon = 2)
julia> D_out[1]
```

<a id='MultivariateAnomalies.init_dist_matrix' href='#MultivariateAnomalies.init_dist_matrix'>#</a>
**`MultivariateAnomalies.init_dist_matrix`** &mdash; *Function*.



```
init_dist_matrix(data)
init_dist_matrix(data, training_data)
```

initialize a `D_out` object for `dist_matrix!()`.

<a id='MultivariateAnomalies.knn_dists' href='#MultivariateAnomalies.knn_dists'>#</a>
**`MultivariateAnomalies.knn_dists`** &mdash; *Function*.



```
knn_dists(D, k::Int, temp_excl::Int = 5)
```

returns the k-nearest neighbors of a distance matrix `D`. Excludes `temp_excl` (default: `temp_excl = 5`) distances from the main diagonal of `D` to be also nearest neighbors.

```jlcon
julia> dc = randn(20, 4,3)
julia> D = dist_matrix(dc, space = 2)
julia> knn_dists_out = knn_dists(D, 3, 1)
julia> knn_dists_out[5] # distances
julia> knn_dists_out[4] # indices
```

<a id='MultivariateAnomalies.knn_dists!' href='#MultivariateAnomalies.knn_dists!'>#</a>
**`MultivariateAnomalies.knn_dists!`** &mdash; *Function*.



```
knn_dists!(knn_dists_out, D, temp_excl::Int = 5)
```

returns the k-nearest neighbors of a distance matrix `D`. Similar to `knn_dists()`, but uses preallocated input object `knn_dists_out`, initialized with `init_knn_dists()`. Please note that the number of nearest neighbors `k` is not necessary, as it is already determined by the `knn_dists_out` object.

```jlcon
julia> dc = randn(20, 4,3)
julia> D = dist_matrix(dc, space = 2)
julia> knn_dists_out = init_knn_dists(dc, 3)
julia> knn_dists!(knn_dists_out, D)
julia> knn_dists_out[5] # distances
julia> knn_dists_out[4] # indices
```

<a id='MultivariateAnomalies.init_knn_dists' href='#MultivariateAnomalies.init_knn_dists'>#</a>
**`MultivariateAnomalies.init_knn_dists`** &mdash; *Function*.



```
init_knn_dists(T::Int, k::Int)
init_knn_dists(datacube::AbstractArray, k::Int)
```

initialize a preallocated `knn_dists_out` object. `k`is the number of nerarest neighbors, `T` the number of time steps (i.e. size of the first dimension) or a multidimensional `datacube`.

<a id='MultivariateAnomalies.kernel_matrix' href='#MultivariateAnomalies.kernel_matrix'>#</a>
**`MultivariateAnomalies.kernel_matrix`** &mdash; *Function*.



```
kernel_matrix(D::AbstractArray, σ::Float64 = 1.0[, kernel::ASCIIString = "gauss", dimension::Int64 = 1])
```

compute a kernel matrix out of distance matrix `D`, given `σ`. Optionally normalized by the `dimension`, if `kernel = "normalized_gauss"`. compute `D` with `dist_matrix()`.

```jlcon
julia> dc = randn(20, 4,3)
julia> D = dist_matrix(dc, space = 2)
julia> K = kernel_matrix(D, 2.0)
```

<a id='MultivariateAnomalies.kernel_matrix!' href='#MultivariateAnomalies.kernel_matrix!'>#</a>
**`MultivariateAnomalies.kernel_matrix!`** &mdash; *Function*.



```
kernel_matrix!(K, D::AbstractArray, σ::Float64 = 1.0[, kernel::ASCIIString = "gauss", dimension::Int64 = 1])
```

compute a kernel matrix out of distance matrix `D`. Similar to `kernel_matrix()`, but with preallocated Array K (`K = similar(D)`) for output.

```jlcon
julia> dc = randn(20, 4,3)
julia> D = dist_matrix(dc, space = 2)
julia> kernel_matrix!(D, D, 2.0) # overwrites distance matrix
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
- [`MultivariateAnomalies.detectAnomalies`](DetectionAlgorithms.md#MultivariateAnomalies.detectAnomalies)
- [`MultivariateAnomalies.detectAnomalies!`](DetectionAlgorithms.md#MultivariateAnomalies.detectAnomalies!)
- [`MultivariateAnomalies.dist_matrix`](DistDensity.md#MultivariateAnomalies.dist_matrix)
- [`MultivariateAnomalies.dist_matrix!`](DistDensity.md#MultivariateAnomalies.dist_matrix!)
- [`MultivariateAnomalies.getParameters`](DetectionAlgorithms.md#MultivariateAnomalies.getParameters)
- [`MultivariateAnomalies.get_MedianCycle`](FeatureExtraction.md#MultivariateAnomalies.get_MedianCycle)
- [`MultivariateAnomalies.get_MedianCycle!`](FeatureExtraction.md#MultivariateAnomalies.get_MedianCycle!)
- [`MultivariateAnomalies.get_MedianCycles`](FeatureExtraction.md#MultivariateAnomalies.get_MedianCycles)
- [`MultivariateAnomalies.get_quantile_scores`](Scores.md#MultivariateAnomalies.get_quantile_scores)
- [`MultivariateAnomalies.get_quantile_scores!`](Scores.md#MultivariateAnomalies.get_quantile_scores!)
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
- [`MultivariateAnomalies.init_detectAnomalies`](DetectionAlgorithms.md#MultivariateAnomalies.init_detectAnomalies)
- [`MultivariateAnomalies.init_dist_matrix`](DistDensity.md#MultivariateAnomalies.init_dist_matrix)
- [`MultivariateAnomalies.init_knn_dists`](DistDensity.md#MultivariateAnomalies.init_knn_dists)
- [`MultivariateAnomalies.kernel_matrix`](DistDensity.md#MultivariateAnomalies.kernel_matrix)
- [`MultivariateAnomalies.kernel_matrix!`](DistDensity.md#MultivariateAnomalies.kernel_matrix!)
- [`MultivariateAnomalies.knn_dists`](DistDensity.md#MultivariateAnomalies.knn_dists)
- [`MultivariateAnomalies.knn_dists!`](DistDensity.md#MultivariateAnomalies.knn_dists!)
- [`MultivariateAnomalies.mw_COR`](FeatureExtraction.md#MultivariateAnomalies.mw_COR)
- [`MultivariateAnomalies.mw_VAR`](FeatureExtraction.md#MultivariateAnomalies.mw_VAR)
- [`MultivariateAnomalies.sMSC`](FeatureExtraction.md#MultivariateAnomalies.sMSC)

