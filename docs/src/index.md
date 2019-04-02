# MultivariateAnomalies.jl 

*A julia package for detecting multivariate anomalies.*

*Keywords: Novelty detection, Anomaly Detection, Outlier Detection, Statistical Process Control, Process Monitoring*

**Please cite this package as:**
Flach, M., Gans, F., Brenning, A., Denzler, J., Reichstein, M., Rodner, E., Bathiany, S., Bodesheim, P., Guanche, Y., Sippel, S., and Mahecha, M. D. (2017): Multivariate anomaly detection for Earth observations: a comparison of algorithms and feature extraction techniques, Earth Syst. Dynam., 8, 677-696, [doi:10.5194/esd-8-677-2017](https://doi.org/10.5194/esd-8-677-2017).

## Requirements

- Julia >= `0.7`
- Julia packages `Distances`, `Combinatorics`, `LinearAlgebra`, and `LIBSVM`.

## Installation

- add the package: ```]add MultivariateAnomalies```

## Package Features

- Detect anomalies in your data with easy to use [high level functions](man/HighLevelFunctions.md) or individual [anomaly detection algorithms](man/DetectionAlgorithms.md)

- [Feature Extraction](man/Preprocessing.md): Preprocess your data by extracting relevant features

- [Similarities and Dissimilarities](man/DistancesDensity.md): Compute distance matrices, kernel matrices and k-nearest neighbor objects.

- [Postprocessing](man/Postprocessing.md): Postprocess your anomaly scores, by computing their quantiles or combinations of several algorithms (ensembles).

- [AUC](man/AUC.md): Compute the area under the curve as external evaluation metric of your scores.

- [Online Algorithms](man/OnlineAlgorithms.md): Algorithms tuned for little memory allocation.

## Using the Package

For a quick start it might be useful to start with the [high level functions](man/HighLevelFunctions.md) for detecting anomalies. They can be used in highly automized way. 

## Input Data

*MultivariateAnomalies.jl* assumes that observations/samples/time steps are stored along the first dimension of the data array (rows of a matrix) with the number of observations `T = size(data, 1)`. Variables/attributes are stored along the last dimension `N` of the data array (along the columns of a matrix) with the number of variables `VAR = size(data, N)`. The implemented anomaly detection algorithms return anomaly scores indicating which observation(s) of the data are anomalous.

## Authors
<img align="right" src="img/MPG_Minerva.png" alt="Minerva" width="75"/> 
The package was implemented by Milan Flach and Fabian Gans, Max Planck Institute for Biogeochemistry, Department Biogeochemical Integration, Research group for Empirical Inference of the Earth System, Jena. 


## Index

```@index
Pages = ["man/Preprocessing.md", "man/DetectionAlgorithms.md", "man/Postprocessing.md", "man/AUC.md", "man/DistancesDensity.md", "man/OnlineAlgorithms.md"]
```
