## High Level Anomaly Detection Algorithms

We provide high-level convenience functions for detecting the anomalies. Namely the pair of 

`P = getParameters(algorithms, training_data)` 
and
`detectAnomalies(testing_data, P)`

sets standard choices of the Parameters `P` and hands the parameters as well as the algorithms choice over to detect the anomalies. 

Currently supported algorithms include Kernel Density Estimation (`algorithms = ["KDE"]`), Recurrences (`"REC"`), k-Nearest Neighbors algorithms (`"KNN-Gamma"`, `"KNN-Delta"`), Hotelling's $T^2$ (`"T2"`), Support Vector Data Description (`"SVDD"`) and Kernel Null Foley Summon Transform (`"KNFST"`). With `getParameters()` it is also possible to compute output scores of multiple algorithms at once (`algorihtms = ["KDE", "T2"]`), quantiles of the output anomaly scores (`quantiles = true`) and ensembles of the selected algorithms (e.g. `ensemble_method = "mean"`). 


### Functions

```@docs
getParameters
detectAnomalies
detectAnomalies!
init_detectAnomalies
```

## Index

```@index
Pages = ["HighLevelFunctions.md"]
```