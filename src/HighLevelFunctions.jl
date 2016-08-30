type PARAMS
  algorithms::Array{ASCIIString,1}
  training_data::Array{Float64,2}
  dist::ASCIIString
  K_sigma::Float64
  KNN_k::Int64
  REC_varepsilon::Float64
  Q::Array{Float64,2} #covariance matrix
  mv::Array{Float64,2} # mean vector
  SVDD_nu::Float64
  SVDD_model
  KNFST_model
  # initialisation
  D::Tuple{Array{Float64,2},Array{Float64,2},Array{Float64,2}}
  K::Array{Float64,2}
  # algorithms
  KDE::Array{Float64,1}
  REC::Array{Float64,1}
  KNN #::Tuple{Int64,Array{Int64,1},Array{Float64,1},Array{Int64,2},Array{Float64,2}}
  KNN_Gamma::Array{Float64,1}
  KNN_Delta# ::Tuple{Array{Float64,1},Array{Float64,2},Array{Float64,2}}
  T2 #::Tuple{Array{Float64,1},Array{Float64,2},Array{Float64,2},Array{Float64,2},Array{Float64,2}}
  D_train::Tuple{Array{Float64,2},Array{Float64,2},Array{Float64,2}}
  D_test::Tuple{Array{Float64,2},Array{Float64,2},Array{Float64,2},Array{Float64,2},Array{Float64,2}}
  KNFST
  SVDD # ::Tuple{Array{Any,1},Array{Float64,2},Array{LIBSVM.SVMNode,2},Array{Ptr{LIBSVM.SVMNode},1}}
  data
  temp_excl::Int64
  ensemble_method::ASCIIString
  ensemble
  quantiles::Bool
end

function reshape_dims_except_N{tp, N}(datacube::AbstractArray{tp,N})
  dims = 1
    for i = 1:(N-1) # multiply dimensions except the last one and save as dims
        dims = size(datacube, i) * dims
    end
    X = reshape(datacube, dims, size(datacube, N))
    return(X)
end

function ispartof(needle::Array{ASCIIString,1}, haystack::Array{ASCIIString,1})
  partof = fill(false, size(haystack, 1))
  for i = 1:size(needle, 1)
    partof[needle[i] .== haystack] = true
  end
  return(partof)
end

"""
    getParameters(algorithms::Array{ASCIIString,1} = ["REC", "KDE"], training_data::AbstractArray{tp, 2} = [NaN NaN])

return an object of type PARAMS, given the `algorithms` and some `training_data`.

# Arguments
- `dist::ASCIIString = "Euclidean"`
- `sigma_quantile::Float64 = 0.5`
- 'varepsilon_quantile::Float64 = NaN`
- `k_perc::Float64 = 0.05`
- `nu::Float64 = 0.2`
- `temp_excl::Int64 = 0`
- `ensemble_method = "None"`
- `quantiles = false`
"""

function getParameters{tp}(algorithms::Array{ASCIIString,1} = ["REC", "KDE"], training_data::AbstractArray{tp, 2} = [NaN NaN]; dist::ASCIIString = "Euclidean", sigma_quantile::Float64 = 0.5, varepsilon_quantile::Float64 = NaN, k_perc::Float64 = 0.05, nu::Float64 = 0.2, temp_excl::Int64 = 0, ensemble_method = "None", quantiles = false)
  @assert any(ispartof(algorithms, ["REC", "KDE", "KNN_Gamma", "KNN_Delta", "SVDD", "KNFST", "T2"]))
  T = size(training_data, 1)

  if(length(size(training_data)) > 2) training_data = reshape_dims_except_N(training_data) end

  #Parameters = PARAMS(algorithms, training_data, dist, NaN, 0, NaN, [NaN NaN], [NaN NaN], NaN, NaN, NaN)
    P = PARAMS(algorithms, training_data, "Euclidean"
               , NaN, 0, NaN, [NaN NaN], [NaN NaN], NaN, NaN, NaN
                      , ([NaN NaN], [NaN NaN], [NaN NaN]) #D
                      , [NaN NaN] # K
                      ,  [NaN] # KDE
                      , [NaN] # REC
                      , NaN #KNN
                      , [NaN] # KNN_Gamma
                      , NaN # KNN_Delta
                      , NaN # T2
                      , ([NaN NaN], [NaN NaN], [NaN NaN]) # D_train
                      , ([NaN NaN], [NaN NaN], [NaN NaN], [NaN NaN], [NaN NaN])  # D test
                      , NaN # KNFST
                      , NaN # SVDD
                      , NaN# data
                      , temp_excl, ensemble_method, NaN, quantiles
                         )
  if(any(dist .== ["Mahalanobis", "SqMahalanobis"]) || any(algorithms .== "T2"))
      P.Q = cov(training_data)
      P.mv = mean(training_data, 1)
  end
  if(any(algorithms .== "REC") || any(algorithms .== "KDE") || any(algorithms .== "SVDD") || any(algorithms .== "KNFST"))
    D_train = dist_matrix(training_data, dist = P.dist, Q = P.Q)
    if(isnan(varepsilon_quantile)) varepsilon_quantile = sigma_quantile end
    (P.K_sigma, P.REC_varepsilon) = quantile(pointer_to_array(pointer(D_train), length(D_train)), [sigma_quantile, varepsilon_quantile])
  end
  if(any(algorithms .== "KNN_Gamma") || any(algorithms .== "KNN_Delta"))
    P.KNN_k = Int(ceil(k_perc * T))
  end
  if(any(algorithms .== "SVDD") || any(algorithms .== "KNFST"))
    kernel_matrix!(D_train, D_train, P.K_sigma)
    if(any(algorithms .== "KNFST"))  P.KNFST_model = KNFST_train(D_train) end
    if(any(algorithms .== "SVDD"))   P.SVDD_model = SVDD_train(D_train, nu) end
  end

  return(P)
end


function init_getAnomalies{tp, N}(data::AbstractArray{tp, N}, P::PARAMS)
# initialisation
  T = size(data, 1)
  VARs = size(data, N)
  P.D = init_dist_matrix(data)
  P.K = similar(P.D[1])
  if(any(P.algorithms .== "KDE")) P.KDE = init_KDE(T) end
  if(any(P.algorithms .== "REC")) P.REC = init_REC(T) end


  if(any(P.algorithms .== "KNN_Gamma") || any(P.algorithms .== "KNN_Delta"))
    P.KNN = init_knn_dists(data, P.KNN_k)
    if(any(P.algorithms .== "KNN_Gamma")) P.KNN_Gamma = init_KNN_Gamma(T) end
    if(any(P.algorithms .== "KNN_Delta")) P.KNN_Delta = init_KNN_Delta(T, VARs, P.KNN_k) end
  end

  if(any(P.algorithms .== "T2")) P.T2 =  init_T2(VARs, T) end

  if(any(P.algorithms .== "SVDD") || any(P.algorithms .== "KNFST"))
    P.D_train = init_dist_matrix(P.training_data)
    P.D_test = init_dist_matrix(data, P.training_data)
    if(any(P.algorithms .== "KNFST")) P.KNFST = init_KNFST(T, P.KNFST_model) end
    if(any(P.algorithms .== "SVDD"))  P.SVDD = init_SVDD_predict(T, size(P.training_data, 1)) end
  end

  return(P)
end



function detectAnomalies!{tp, N}(data::AbstractArray{tp, N}, P::PARAMS)
  P.data = data
  allalgorithms = ["KDE", "REC", "KNN_Delta", "KNN_Gamma", "T2", "SVDD", "KNFST"]
  @assert any(ispartof(P.algorithms, allalgorithms))

  if(any(ispartof(P.algorithms, ["KDE", "REC", "KNN_Delta", "KNN_Gamma"])))   # D based
    dist_matrix!(P.D, P.data, dist = P.dist, Q = P.Q)
    if(any(ispartof(P.algorithms, ["KDE"])))
      kernel_matrix!(P.K, P.D[1], P.K_sigma)
      KDE!(P.KDE, P.K)
      broadcast!(*, P.KDE, P.KDE, -1)
    end

     if(any(ispartof(P.algorithms, ["REC"])))
      REC!(P.REC, P.D[1], P.REC_varepsilon, P.temp_excl)
      broadcast!(*, P.REC, P.REC, -1)
    end

    if(any(ispartof(P.algorithms, ["KNN_Gamma", "KNN_Delta"])))
      knn_dists!(P.KNN, P.D[1], P.temp_excl)

      if(any(ispartof(P.algorithms, ["KNN_Gamma"]))) KNN_Gamma!(P.KNN_Gamma, P.KNN) end
      if(any(ispartof(P.algorithms, ["KNN_Delta"]))) KNN_Delta!(P.KNN_Delta, P.KNN, P.data) end

    end
  end

  if(any(ispartof(P.algorithms, ["T2"])))
    T2!(P.T2, P.data, P.Q, P.mv)
  end

  if(any(ispartof(P.algorithms, ["SVDD", "KNFST"])))
    dist_matrix!(P.D_train, P.training_data, dist = P.dist, Q = P.Q)
    dist_matrix!(P.D_test, P.data, P.training_data, dist = P.dist, Q = P.Q)
    kernel_matrix!(P.D_train[1],P.D_train[1], P.K_sigma) # transform distance to kernel matrix
    kernel_matrix!(P.D_test[1], P.D_test[1], P.K_sigma) # transform distance to kernel matrix
    if(any(ispartof(P.algorithms, ["KNFST"])))  KNFST_predict!(P.KNFST, P.KNFST_model, P.D_test[1]) end
    if(any(ispartof(P.algorithms, ["SVDD"])))  SVDD_predict!(P.SVDD, P.SVDD_model, P.D_test[1]) end
  end

  if(P.quantiles)
    if(any(ispartof(P.algorithms, ["T2"])))  P.T2[1] = get_quantile_scores(P.T2[1]) end
    if(any(ispartof(P.algorithms, ["REC"]))) P.REC = get_quantile_scores(P.REC) end
    if(any(ispartof(P.algorithms, ["KDE"]))) P.KDE = get_quantile_scores(P.KDE) end
    if(any(ispartof(P.algorithms, ["SVDD"]))) P.SVDD[2] = get_quantile_scores(P.SVDD[2]) end
    if(any(ispartof(P.algorithms, ["KNFST"]))) P.KNFST[1] = get_quantile_scores(P.KNFST[1]) end
    if(any(ispartof(P.algorithms, ["KNN_Gamma"]))) P.KNN_Gamma = get_quantile_scores(P.KNN_Gamma) end
    if(any(ispartof(P.algorithms, ["KNN_Delta"]))) P.KNN_Delta[1] = get_quantile_scores(P.KNN_Delta[1]) end
  end


  return(P)
end


"""
    detectAnomalies{tp, N}(data::AbstractArray{tp, N}, P::PARAMS)
    detectAnomalies{tp, N}(data::AbstractArray{tp, N}, algorithms::Array{ASCIIString,1} = ["REC", "KDE"]; mean = 0)

detect anomalies, given some Parameter object `P` of type PARAMS. Train the Parameters `P` with `getParameters()` beforehand on some training data.
Without training `P` beforehand, it is also possible to use `detectAnomalies(data, algorithms)` given some algorithms (except SVDD, KNFST).
Some default parameters are used in this case to initialize `P` internally.
"""

function detectAnomalies{tp, N}(data::AbstractArray{tp, N}, P::PARAMS)
  init_getAnomalies(data, P)
  detectAnomalies!(data, P)
  L = length(P.algorithms)
  if(L == 1) return(getfield(P, parse(algorithms[1]))) end
  if(L == 2) return(getfield(P, parse(algorithms[1])), getfield(P, parse(algorithms[2]))) end
  if(L == 3) return(getfield(P, parse(algorithms[1])), getfield(P, parse(algorithms[2])), getfield(P, parse(algorithms[3])) ) end
  if(L == 4) return(getfield(P, parse(algorithms[1])), getfield(P, parse(algorithms[2])), getfield(P, parse(algorithms[3])), getfield(P, parse(algorithms[4])) ) end
  if(L == 5) return(getfield(P, parse(algorithms[1])), getfield(P, parse(algorithms[2])), getfield(P, parse(algorithms[3])), getfield(P, parse(algorithms[4])),  getfield(P, parse(algorithms[5]))) end
  if(L == 6) return(getfield(P, parse(algorithms[1])), getfield(P, parse(algorithms[2])), getfield(P, parse(algorithms[3])), getfield(P, parse(algorithms[4])),  getfield(P, parse(algorithms[5])),   getfield(P, parse(algorithms[6]))) end
  if(L == 7) return(getfield(P, parse(algorithms[1])), getfield(P, parse(algorithms[2])), getfield(P, parse(algorithms[3])), getfield(P, parse(algorithms[4])),  getfield(P, parse(algorithms[5])),   getfield(P, parse(algorithms[6])), getfield(P, parse(algorithms[7]))) end
end


function detectAnomalies{tp, N}(data::AbstractArray{tp, N}, algorithms::Array{ASCIIString,1} = ["REC", "KDE"]; mean = 0)
  @assert !any(ispartof(algorithms, ["SVDD", "KNFST"]))
   Q = cov(reshape_dims_except_N(data))
   if(mean == 0) meanvec = zeros(Float64, 1, size(data, N))
   else  meanvec = mean(data, 1) end
   D = dist_matrix(data)
   sigma =  median(pointer_to_array(pointer(D), length(D)))
   P = PARAMS(algorithms, [NaN NaN], dist, sigma # sigma
             , Int(ceil(0.05 * size(data, 1))) # k
             , sigma # varepsilon
             , Q
             , meanvec, 0.2, NaN, NaN
                      , ([NaN NaN], [NaN NaN], [NaN NaN]) #D
                      , [NaN NaN] # K
                      ,  [NaN] # KDE
                      , [NaN] # REC
                      , NaN #KNN
                      , [NaN] # KNN_Gamma
                      , NaN # KNN_Delta
                      , NaN # T2
                      , ([NaN NaN], [NaN NaN], [NaN NaN]) # D_train
                      , ([NaN NaN], [NaN NaN], [NaN NaN], [NaN NaN], [NaN NaN])  # D test
                      , NaN # KNFST
                      , NaN # SVDD
                      , data # data
                      , 0, "None", NaN, false
                         )
  init_getAnomalies(data, P)
  detectAnomalies!(data, P)
  L = length(P.algorithms)
  if(L == 1) return(getfield(P, parse(algorithms[1]))) end
  if(L == 2) return(getfield(P, parse(algorithms[1])), getfield(P, parse(algorithms[2]))) end
  if(L == 3) return(getfield(P, parse(algorithms[1])), getfield(P, parse(algorithms[2])), getfield(P, parse(algorithms[3])) ) end
  if(L == 4) return(getfield(P, parse(algorithms[1])), getfield(P, parse(algorithms[2])), getfield(P, parse(algorithms[3])), getfield(P, parse(algorithms[4])) ) end
  if(L == 5) return(getfield(P, parse(algorithms[1])), getfield(P, parse(algorithms[2])), getfield(P, parse(algorithms[3])), getfield(P, parse(algorithms[4])),  getfield(P, parse(algorithms[5]))) end
  #if(L == 6) return(getfield(P, parse(algorithms[1])), getfield(P, parse(algorithms[2])), getfield(P, parse(algorithms[3])), getfield(P, parse(algorithms[4])),  getfield(P, parse(algorithms[5])),   getfield(P, parse(algorithms[6]))) end
  #if(L == 7) return(getfield(P, parse(algorithms[1])), getfield(P, parse(algorithms[2])), getfield(P, parse(algorithms[3])), getfield(P, parse(algorithms[4])),  getfield(P, parse(algorithms[5])),   getfield(P, parse(algorithms[6])), getfield(P, parse(algorithms[7]))) end
end


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
