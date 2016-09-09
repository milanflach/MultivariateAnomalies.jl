# MultivariateAnomalies.jl 

*A julia package for detecting multivariate anomalies.*

*Keywords: Novelty detection, Anomaly Detection, Outlier Detection, Statistical Process Control*

    Please cite this package as ...

## Requirements

- Julia `0.4`
- `Distances`, `MultivariateStats`
-  latest `LIBSVM` branch via:
`Pkg.clone("https://github.com/milanflach/LIBSVM.jl.git");`
`Pkg.checkout("LIBSVM", "mutating_versions");`
`Pkg.build("LIBSVM")`

## Package Features

- Detect anomalies in your data with 

    - easy to use [high level functions]("man/HighLevelFunctions.md")
    - individual [anomaly detection algorithms]("manAnomalyDetection.md")

- [Feature Extraction](man/Preprocessing.md): Preprocess your data by extracting relevant features

- [Similarities and Dissimilarities](man/DistancesDensity.md): Compute distance matrices, kernel matrices and k-nearest neighbor objects.

- [Postprocessing](man/Postprocessing.md): Postprocess your anomaly scores, by computing their quantiles or combinations of several algorithms (ensembles).

- [AUC](man/AUC.md): Compute the area under the curve as external evaluation metric of your scores.

## Using the Package

For a quick start it might be useful to start with the [high-level convenience functions]("man/HighLevelFunctions.md") for detecting anomalies. They can be used in highly automized way. 

## Input Data

*MultivariateAnomalies.jl* assumes that observations/samples/time steps are stored along the first dimension of the data array (rows of a matrix) with the number of observations `T = size(data, 1)`. Variables/attributes are stored along the last dimension `N` of the data array (along the columns of a matrix) with the number of variables `VAR = size(data, N)`. The implemented anomaly detection algorithms return anomaly scores indicating which observation(s) of the data are anomalous.

## Index

```@index
Pages = ["man/Preprocessing.md", "man/DetectionAlgorithms.md", "man/Postprocessing.md", "man/AUC.md", "man/DistancesDensity.md"]
```