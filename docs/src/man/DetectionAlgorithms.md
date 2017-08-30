## Anomaly Detection Algorithms

Most of the anomaly detection algorithms below work on a distance/similarity matrix `D` or a kernel/dissimilarity matrix `K`. They can be comuted using the functions provided [here](DistancesDensity.md).

Currently supported algorithms include

- Recurrences (REC)
- Kernel Density Estimation (KDE)
- Hotelling's $T^2$ (Mahalanobis distance) (T2)
- two k-Nearest Neighbor approaches (KNN-Gamma, KNN-Delta)  
- Univariate Approach (UNIV)
- Support Vector Data Description (SVDD)
- Kernel Null Foley Summon Transform (KNFST)


## Functions

### Recurrences

```@docs
REC
REC!
init_REC
```

### Kernel Density Estimation

```@docs
KDE
KDE!
init_KDE
```

### Hotelling's T<sup>2</sup>
```@docs
T2
T2!
init_T2
```

### k-Nearest Neighbors
```@docs
KNN_Gamma
KNN_Gamma!
init_KNN_Gamma
KNN_Delta
KNN_Delta!
init_KNN_Delta
```

### Univariate Approach
```@docs
UNIV
UNIV!
init_UNIV
```

### Support Vector Data Description
```@docs
SVDD_train
SVDD_predict
```

### Kernel Null Foley Summon Transform
```@docs
KNFST_train
KNFST_predict
KNFST_predict!
init_KNFST  
```

### Distance to some Centers

```@docs
Dist2Centers
```

## Index

```@index
Pages = ["DetectionAlgorithms"]
```