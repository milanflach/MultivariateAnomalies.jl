## Distance, Kernel Matrices and k-Nearest Neighbours 

Compute distance matrices (similarity matrices) and convert them into kernel matrices or k-nearest neighbor objects.


## Distance/Similarity Matrices

A distance matrix `D` consists of pairwise distances $d()$computed with some metrix (e.g. Euclidean):
  
$D = d(X_{t_i}, X_{t_j})$

i.e. the distance between vector $X$ of observation $t_i$ and $t_j$ for all observations $t_i,t_j = 1 \ldots T$.

#### Functions

```@docs
dist_matrix
dist_matrix!
init_dist_matrix
```

## k-Nearest Neighbor Objects

k-Nearest Neighbor objects return the k nearest points and their distance out of a distance matrix `D`.

#### Functions

```@docs
knn_dists
knn_dists!
init_knn_dists
```

## Kernel Matrices (Dissimilarities)

A distance matrix `D` can be converted into a kernel matrix `K`, i.e. by computing pairwise dissimilarities using Gaussian kernels centered on each datapoint. 

$K= exp(-0.5 \cdot D \cdot \sigma^{-2})$

#### Functions

```@docs
kernel_matrix
kernel_matrix!
```

## Index

```@index
Pages = ["DistancesDensity.md"]
```