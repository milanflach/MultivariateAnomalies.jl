
<a id='Feature-Extraction-Techniques-1'></a>

## Feature Extraction Techniques


Extract the relevant inforamtion out of your data and use them as input feature for the anomaly detection algorithms.


<a id='Functions-1'></a>

## Functions

<a id='MultivariateAnomalies.sMSC' href='#MultivariateAnomalies.sMSC'>#</a>
**`MultivariateAnomalies.sMSC`** &mdash; *Function*.



```
sMSC(datacube, cycle_length)
```

subtract the median seasonal cycle from the datacube given the length of year `cycle_length`.

**Examples**

```jlcon
julia> dc = hcat(rand(193) + 2* sin(0:pi/24:8*pi), rand(193) + 2* sin(0:pi/24:8*pi))
julia> sMSC_dc = sMSC(dc, 48)
```

<a id='MultivariateAnomalies.globalPCA' href='#MultivariateAnomalies.globalPCA'>#</a>
**`MultivariateAnomalies.globalPCA`** &mdash; *Function*.



```
globalPCA{tp, N}(datacube::Array{tp, N}, expl_var::Float64 = 0.95)
```

return an orthogonal subset of the variables, i.e. the last dimension of the datacube. A Principal Component Analysis is performed on the entire datacube, explaining at least `expl_var` of the variance.

<a id='MultivariateAnomalies.globalICA' href='#MultivariateAnomalies.globalICA'>#</a>
**`MultivariateAnomalies.globalICA`** &mdash; *Function*.



```
globalICA(datacube::Array{tp, 4}, mode = "expl_var"; expl_var::Float64 = 0.95, num_comp::Int = 3)
```

perform an Independent Component Analysis on the entire 4-dimensional datacube either by (`mode = "num_comp"`) returning num_comp number of independent components or (`mode = "expl_var"`) returning the number of components which is necessary to explain expl_var of the variance, when doing a Prinicpal Component Analysis before.

<a id='MultivariateAnomalies.TDE' href='#MultivariateAnomalies.TDE'>#</a>
**`MultivariateAnomalies.TDE`** &mdash; *Function*.



```
TDE{tp}(datacube::Array{tp, 4}, ΔT::Integer, DIM::Int = 3)
TDE{tp}(datacube::Array{tp, 3}, ΔT::Integer, DIM::Int = 3)
```

returns an embedded datacube by concatenating lagged versions of the 2-, 3- or 4-dimensional datacube with `ΔT` time steps in the past up to dimension `DIM` (presetting: `DIM = 3`)

```jldoctest julia> dc = randn(50,3) julia> TDE(dc, 3, 2)

<a id='MultivariateAnomalies.mw_VAR' href='#MultivariateAnomalies.mw_VAR'>#</a>
**`MultivariateAnomalies.mw_VAR`** &mdash; *Function*.



```
mw_VAR{tp,N}(datacube::Array{tp,N}, windowsize::Int = 10)
```

compute the variance in a moving window along the first dimension of the datacube (presetting: `windowsize = 10`). Accepts N dimensional datacubes.

```jlcon
julia> dc = randn(50,3,3,3)
julia> mw_VAR(dc, 15)
```

<a id='MultivariateAnomalies.mw_COR' href='#MultivariateAnomalies.mw_COR'>#</a>
**`MultivariateAnomalies.mw_COR`** &mdash; *Function*.



```
mw_COR{tp}(datacube::Array{tp, 4}, windowsize::Int = 10)
```

compute the correlation in a moving window along the first dimension of the datacube (presetting: `windowsize = 10`). Accepts 4-dimensional datacubes.

<a id='MultivariateAnomalies.EWMA' href='#MultivariateAnomalies.EWMA'>#</a>
**`MultivariateAnomalies.EWMA`** &mdash; *Function*.



```
EWMA(dat,  λ)
```

Compute the exponential weighted moving average (EWMA) with the weighting parameter `λ` between 0 (full weighting) and 1 (no weighting) in the first dimension of `dat`. Supports N-dimensional Arrays.

```jlcon
julia> dc = rand(100,3,2)
julia> ewma_dc = EWMA(dc, 0.1)
```

<a id='MultivariateAnomalies.EWMA!' href='#MultivariateAnomalies.EWMA!'>#</a>
**`MultivariateAnomalies.EWMA!`** &mdash; *Function*.



```
EWMA!(Z, dat,  λ)
```

use a preallocated output Z. `Z = similar(dat)` or `dat = dat` for overwriting itself.

**Examples**

```jlcon
julia> dc = rand(100,3,2)
julia> EWMA!(dc, dc, 0.1)
```

<a id='MultivariateAnomalies.get_MedianCycles' href='#MultivariateAnomalies.get_MedianCycles'>#</a>
**`MultivariateAnomalies.get_MedianCycles`** &mdash; *Function*.



```
get_MedianCycles(datacube, cycle_length::Int = 46)
```

returns the median annual cycle of a datacube, given the length of the annual cycle (presetting: `cycle_length = 46`). The datacube can be 2, 3, 4-dimensional, time is stored along the first dimension.

**Examples**

```jlcon
julia> dc = hcat(rand(193) + 2* sin(0:pi/24:8*pi), rand(193) + 2* sin(0:pi/24:8*pi))
julia> cycles = get_MedianCycles(dc, 48)
```

<a id='MultivariateAnomalies.get_MedianCycle' href='#MultivariateAnomalies.get_MedianCycle'>#</a>
**`MultivariateAnomalies.get_MedianCycle`** &mdash; *Function*.



```
get_MedianCycle(dat::Array{tp,1}, cycle_length::Int = 46)
```

returns the median annual cycle of a one dimensional data array, given the length of the annual cycle (presetting: cycle_length = 46). Can deal with some NaN values.

**Examples**

```jlcon
julia> dat = rand(193) + 2* sin(0:pi/24:8*pi)
julia> dat[100] = NaN
julia> cycles = get_MedianCycle(dat, 48)
```

<a id='MultivariateAnomalies.get_MedianCycle!' href='#MultivariateAnomalies.get_MedianCycle!'>#</a>
**`MultivariateAnomalies.get_MedianCycle!`** &mdash; *Function*.



```
get_MedianCycle!(init_MC, dat::Array{tp,1})
```

Memory efficient version of get_MedianCycle, returning the median cycle in `init_MC[3]`. The init_MC object should be created with init_MedianCycle. Can deal with some NaN values.

**Examples**

```jlcon
julia> dat = rand(193) + 2* sin(0:pi/24:8*pi)
julia> dat[100] = NaN
julia> init_MC = init_MedianCycle(dat, 48)
julia> get_MedianCycle!(init_MC, dat)
julia> init_MC[3]
```

<a id='MultivariateAnomalies.init_MedianCycle' href='#MultivariateAnomalies.init_MedianCycle'>#</a>
**`MultivariateAnomalies.init_MedianCycle`** &mdash; *Function*.



```
init_MedianCycle(dat::Array{tp}, cycle_length::Int = 46)
init_MedianCycle(temporal_length::Int[, cycle_length::Int = 46])
```

initialises an init_MC object to be used as input for get_MedianCycle!. Input is either some sample data or the temporal lenght of the expected input vector and the length of the annual cycle (presetting: cycle_length = 46)


<a id='Index-1'></a>

## Index

- [`MultivariateAnomalies.EWMA`](FeatureExtraction.md#MultivariateAnomalies.EWMA)
- [`MultivariateAnomalies.EWMA!`](FeatureExtraction.md#MultivariateAnomalies.EWMA!)
- [`MultivariateAnomalies.KDE`](DetectionAlgorithms.md#MultivariateAnomalies.KDE)
- [`MultivariateAnomalies.KDE!`](DetectionAlgorithms.md#MultivariateAnomalies.KDE!)
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

