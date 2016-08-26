"""
    detectAnomalies{tp, N}(data::AbstractArray{tp, N}; algorithm = "KDE", dist = "Euclidean", sigma::Float64 = 1.0, k::Int = 2, nu::Float64 = 0.2[, ...])

detect Anomalies with the detection algorihtm `algorithm`. High output scores are more likely to be anoamlies than low scores.

# Arguments

- `algorithm`: the detection algorithm, one of `["KDE", "REC", "KNN_Delta", "KNN_Gamma", "T2", "SVDD", "KNFST"]``
- `dist`: the distance measure used for all algorihtms except `T2`. Can be one choice of `dist_matrix()`.
- `sigma`: weighting parameter for `kernel_matrix()`. Used for algorithms `KDE`, `SVDD`, `KNFST``
- `k`: number of nearest neighbors `k`. Used for the algorithms `KNN-Gamma`, `KNN-Delta`.
- `nu`: maximum possible percentage of outliers in `training_data`. Used for algorihtm `SVDD`.
- `varepsilon`: `rec_threshold` used for algorithm `REC`. By default the same as `sigma`.
- `Q`: covariance matrix. Used for `dist = "Mahalanobis"` or `SqMahalanobis` and for the algorihtm `T2`.
- `meanvector`: mean vector of the `data`. Used for the algorithm `T2`.
- `training_data`: training data for the algorihtms `KNFST` and `SVDD`
"""

function detectAnomalies{tp, N}(data::AbstractArray{tp, N}; algorithm::ASCIIString = "KDE", dist::ASCIIString = "Euclidean"
                                , sigma::Float64 = 1.0, k::Int = 2, nu::Float64 = 0.2, varepsilon::Float64 = NaN, Q = NaN, meanvector = NaN
                                , training_data = NaN)
  @assert any(algorithm .== ["KDE", "REC", "KNN_Delta", "KNN_Gamma", "T2", "SVDD", "KNFST"])
  if(isnan(meanvector)) meanvector = mean(data, 1) end

  if(any(algorithm .== ["KDE", "REC", "KNN_Delta", "KNN_Gamma"]))   # D based
    if(isnan(Q) && any(dist .== ["Mahalanobis","SqMahalanobis"]))
        if(isnan(training_data)) Q = cov(data); print("Warning: Covariance matrix not given beforehand, but needed for dist = $dist. Estimated using data.")
        else Q = cov(training_data) end
    end
    D = dist_matrix(data, dist = dist, Q = Q)
    K = kernel_matrix(D, sigma)

    if(algorithm == "KDE") return(-KDE(K)) end
    if(isnan(varepsilon)) varepsilon = sigma  end
    if(algorithm == "REC") return(-REC(D, varepsilon, temp_excl)) end

    if(any(algorithm .== ["KNN_Gamma", "KNN_Delta"])) knn_dists_out = knn_dists(D, k, temp_excl)     # KNN based (also D based)

      if(algorithm == "KNN_Gamma") return(KNN_Gamma(knn_dists_out)) end
      if(algorithm == "KNN_Delta") return(KNN_Delta(knn_dists_out, data)) end

    end
  end

  if(algorithm == "T2")
    if(isnan(Q))
        if(isnan(training_data)) Q = cov(data); print("Warning: Covariance matrix not given beforehand, but needed for $algorithm. Estimated using data.")
          else Q = cov(training_data) end
    end
    return(T2(data, Q, meanvector))
  end

  if(any(algorithm .== ["SVDD", "KNFST"]))
    if(isnan(training_data))
        print("Warning: training_data not specified, but algorihtm $algorithm requires training. Used data for training and testing, which is not recommended")
        training_data = data
    end
    if(isnan(Q) && any(dist .== ["Mahalanobis","SqMahalanobis"])) Q = cov(training_data) end
    D_train = dist_matrix(training_data, dist = dist, Q = Q)
    D_test = dist_matrix(data, training_data, dist = dist, Q = Q)
    kernel_matrix!(D_test, D_test, sigma)
    kernel_matrix!(D_train, D_train, sigma)
    if(algorithm == "KNFST") return(KNFST_predict(KNFST_train(D_train), D_test))[1] end
    if(algorithm == "SVDD") return(SVDD_predict(SVDD_train(D_train, nu), D_test)[2]) end
  end
end
