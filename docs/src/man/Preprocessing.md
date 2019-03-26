## Feature Extraction Techniques

Extract the relevant inforamtion out of your data and use them as input feature for the anomaly detection algorithms.

## Dimensionality Reduction

For dimensionality reduction, we would like to point to the package *MultivariateStats.jl*. Several techniques are implemented there, e.g.

- Principal Component Analysis (PCA)
- Independent Component Analysis (ICA)


## Seasonality

When dealing with time series, i.e. the observations are time steps, it might be important to remove or get robust estimates of the mean seasonal cycles. This is implemended by

- subtracting the median seasonal cycle (sMSC) and
- getting the median seasonal cycle (get_MedianCycles)

#### Functions

```@docs
sMSC
get_MedianCycles
get_MedianCycle
get_MedianCycle!
init_MedianCycle
```

## Exponential Weighted Moving Average

One option to reduce the noise level in the data and detect more 'significant' anomalies is computing an exponential weighted moving average (EWMA)

#### Function

```@docs
EWMA
EWMA!
```

## Time Delay Embedding

Increase the feature space (Variabales) with lagged observations. 

#### Function

```@docs
TDE
```

## Moving Window Features

include the variance (mw_VAR) and correlations (mw_COR) in a moving window along the first dimension of the data.

#### Functions

```@docs
mw_VAR
mw_VAR!
mw_COR
mw_AVG
mw_AVG!
```

## Index

```@index
Pages = ["Preprocessing.md"]
```