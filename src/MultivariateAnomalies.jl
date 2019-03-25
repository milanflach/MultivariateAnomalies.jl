__precompile__(true)

module MultivariateAnomalies
#######################
  #import MultivariateStats
  import Combinatorics
  using LinearAlgebra
  using Distances
  using LIBSVM
  using Base.Cartesian
  using Statistics
  using StatsBase

  # DistDensity
  export
    dist_matrix,
    dist_matrix!,
    init_dist_matrix,
    knn_dists,
    init_knn_dists,
    knn_dists!,
    kernel_matrix,
    kernel_matrix!,
# Detection Algorithms
    REC,
    init_REC,
    REC!,
    KDE,
    init_KDE,
    KDE!,
    init_T2,
    T2,
    T2!,
    KNN_Gamma,
    init_KNN_Gamma,
    KNN_Gamma!,
    KNN_Delta,
    KNN_Delta!,
    init_KNN_Delta,
    UNIV,
    UNIV!,
    init_UNIV,
    SVDD_train,
    SVDD_predict,
    KNFST_train,
    KNFST_predict,
    KNFST_predict!,
    init_KNFST,
    Dist2Centers,
# FeatureExtraction
    sMSC,
    TDE,
    mw_VAR,
    mw_VAR!,
    mw_AVG,
    mw_AVG!,
    mw_COR,
    EWMA,
    EWMA!,
    get_MedianCycles,
    get_MedianCycle,
    get_MedianCycle!,
    init_MedianCycle,
    mapMovingWindow,
# AUC
    auc,
# Scores
    get_quantile_scores,
    get_quantile_scores!,
    compute_ensemble,
# high level functions
    getParameters,
    detectAnomalies,
    init_detectAnomalies,
    detectAnomalies!,
# online algorithms
    Euclidean_distance!,
    Mahalanobis_distance!,
    SigmaOnline!,
    KDEonline!,
    KNNonline!,
    REConline!

# Distance and Density Estimation
include("DistDensity.jl")
# Multivariate Anomaly Detection Algorithms
include("DetectionAlgorithms.jl")
# AUC computations
include("AUC.jl")
# Feature Extraction techniques
include("FeatureExtraction.jl")
# post processing for Anomaly Scores
include("Scores.jl")
# high level functions
include("HighLevelFunctions.jl")
# online algorithms
include("OnlineAlgorithms.jl")

#######################
end
