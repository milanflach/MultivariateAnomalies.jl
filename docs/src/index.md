# MultivariateAnomalies.jl 

*Decting multivariate anomalies in Julia.*

A package for detecting anomalies in multivariate data.

*Keywords: Novelty detection, Anomaly Detection, Outlier Detection, Statistical Process Control*

    Please cite this package as ...

## Requirements

- Julia `0.4`
- `Distances`, `MultivariateStats`
-  latest `LIBSVM` branch via:
`Pkg.clone("https://github.com/milanflach/LIBSVM.jl.git")`
`Pkg.checkout("LIBSVM", "mutating_versions")`
`Pkg.build("LIBSVM")`

## Package Features

- Extract the relevant features from the data 
```@contents
Pages = ["man/FeatureExtraction.md"]
```
- Detect the anomalies
```@contents
Pages = ["man/DetectionAlgorithms.md"]
```
- Postprocess your anomaly scores, by computing their quantiles or ensembles
```@contents
Pages = ["man/Scores"]
```
- Compute the area under the curve as external evaluation metric
```@contents
Pages = ["man/AUC"]
```

## Using the Package



## Index

```@index
Pages = ["man/FeatureExtraction.md", "man/DetectionAlgorithms.md", "man/Scores", "man/AUC"]
```






