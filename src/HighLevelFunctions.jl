mutable struct PARAMS
  algorithms::Array{String,1}
  training_data::Array{Float64,2}
  dist::String
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
  REC_quantiles::Array{Float64,1}
  KDE_quantiles::Array{Float64,1}
  KNN_Gamma_quantiles::Array{Float64,1}
  KNN_Delta_quantiles::Array{Float64,1}
  T2_quantiles::Array{Float64,1}
  KNFST_quantiles::Array{Float64,1}
  SVDD_quantiles::Array{Float64,1}
  data::Array{Float64,2}
  temp_excl::Int64
  ensemble_method::String
  ensemble
  quantiles::Bool
end

function reshape_dims_except_N(datacube::AbstractArray{tp,N}) where {tp, N}
  dims = 1
    for i = 1:(N-1) # multiply dimensions except the last one and save as dims
        dims = size(datacube, i) * dims
    end
    X = reshape(datacube, dims, size(datacube, N))
    return(X)
end

function ispartof(needle::Array{String,1}, haystack::Array{String,1})
  partof = fill(false, size(haystack, 1))
  for i = 1:size(needle, 1)
    partof[needle[i] .== haystack] .= true
  end
  return(partof)
end

"""
    getParameters(algorithms::Array{String,1} = ["REC", "KDE"], training_data::AbstractArray{tp, 2} = [NaN NaN])

return an object of type PARAMS, given the `algorithms` and some `training_data` as a matrix.

# Arguments
- `algorithms`: Subset of `["REC", "KDE", "KNN_Gamma", "KNN_Delta", "SVDD", "KNFST", "T2"]`
- `training_data`: data for training the algorithms / for getting the Parameters.
- `dist::String = "Euclidean"`
- `sigma_quantile::Float64 = 0.5` (median): quantile of the distance matrix, used to compute the weighting parameter for the kernel matrix (`algorithms = ["SVDD", "KNFST", "KDE"]`)
- `varepsilon_quantile` = `sigma_quantile` by default: quantile of the distance matrix to compute the radius of the hyperball in which the number of reccurences is counted (`algorihtms = ["REC"]`)
- `k_perc::Float64 = 0.05`: percentage of the first dimension of `training_data` to estimmate the number of nearest neighbors (`algorithms = ["KNN-Gamma", "KNN_Delta"]`)
- `nu::Float64 = 0.2`: use the maximal percentage of outliers for `algorithms = ["SVDD"]`
- `temp_excl::Int64 = 0`. Exclude temporal adjacent points from beeing count as recurrences of k-nearest neighbors `algorithms = ["REC", "KNN-Gamma", "KNN_Delta"]`
- `ensemble_method = "None"`: compute an ensemble of the used algorithms. Possible choices (given in `compute_ensemble()`) are "mean", "median", "max" and "min".
- `quantiles = false`: convert the output scores of the algorithms into quantiles.

# Examples
```
julia> using MultivariateAnomalies
julia> training_data = randn(100, 2); testing_data = randn(100, 2);
julia> P = getParameters(["REC", "KDE", "SVDD"], training_data, quantiles = false);
julia> detectAnomalies(testing_data, P)
```
"""
function getParameters(algorithms::Array{String,1} = ["REC", "KDE"], training_data::AbstractArray{tp, N} = [NaN NaN]; dist::String = "Euclidean", sigma_quantile::Float64 = 0.5, varepsilon_quantile::Float64 = NaN, k_perc::Float64 = 0.05, nu::Float64 = 0.2, temp_excl::Int64 = 0, ensemble_method = "None", quantiles = false) where {tp, N}

  allalgorithms = ["KDE", "REC", "KNN_Delta", "KNN_Gamma", "T2", "SVDD", "KNFST"]
  @assert any(ispartof(algorithms, allalgorithms))
  if(length(algorithms) != sum(ispartof(algorithms, allalgorithms)))
    error("one or more of algorithms $algorithms are not within the defined possibilites $allalgorithms")
  end

  T = size(training_data, 1)

  if(length(size(training_data)) > 2) training_data = reshape_dims_except_N(training_data) end

  #Parameters = PARAMS(algorithms, training_data, dist, NaN, 0, NaN, [NaN NaN], [NaN NaN], NaN, NaN, NaN)
    P = PARAMS(algorithms, training_data, "Euclidean"
               , NaN, 0, NaN, [NaN NaN], [NaN NaN], nu, NaN, NaN
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
                              , [NaN], [NaN], [NaN], [NaN], [NaN], [NaN], [NaN] # quantiles
                      , [NaN NaN]# data
                      , temp_excl, ensemble_method, NaN, quantiles
                         )
  if(any(dist .== ["Mahalanobis", "SqMahalanobis"]) || any(algorithms .== "T2"))
      P.Q = cov(training_data)
      P.mv = mean(training_data, dims = 1)
  end
  if(any(algorithms .== "REC") || any(algorithms .== "KDE") || any(algorithms .== "SVDD") || any(algorithms .== "KNFST"))
    D_train = dist_matrix(training_data, dist = P.dist, Q = P.Q)
    if(isnan(varepsilon_quantile)) varepsilon_quantile = sigma_quantile end
    (P.K_sigma, P.REC_varepsilon) = quantile(unsafe_wrap(Array,pointer(D_train), length(D_train)), [sigma_quantile, varepsilon_quantile])
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

"""
    init_detectAnomalies{tp, N}(data::AbstractArray{tp, N}, P::PARAMS)

initialize empty arrays in `P` for detecting the anomalies.
"""
function init_detectAnomalies(data::AbstractArray{tp, N}, P::PARAMS) where {tp, N}
# initialisation
  T = size(data, 1)
  VARs = size(data, N)
  P.data = zeros(tp, T, VARs)
  P.D = init_dist_matrix(data)
  P.K = similar(P.D[1])
  if(any(P.algorithms .== "KDE")) P.KDE = init_KDE(T) end
  if(any(P.algorithms .== "REC")) P.REC = init_REC(T) end


  if(any(P.algorithms .== "KNN_Gamma") || any(P.algorithms .== "KNN_Delta"))
    P.KNN = init_knn_dists(data, P.KNN_k)
    if(any(P.algorithms .== "KNN_Gamma")) P.KNN_Gamma = init_KNN_Gamma(T) end
    if(any(P.algorithms .== "KNN_Delta")) P.KNN_Delta = init_KNN_Delta(T, VARs, P.KNN_k) end
  end

  if(any(P.algorithms .== "T2")) P.T2 =  init_T2(data) end

  if(any(P.algorithms .== "SVDD") || any(P.algorithms .== "KNFST"))
    P.D_train = init_dist_matrix(P.training_data)
    P.D_test = init_dist_matrix(data, P.training_data)
    if(any(P.algorithms .== "KNFST")) P.KNFST = init_KNFST(T, P.KNFST_model) end
    #if(any(P.algorithms .== "SVDD"))  P.SVDD = init_SVDD_predict(T, size(P.training_data, 1)) end
  end

  return(P)
end

"""
    detectAnomalies!{tp, N}(data::AbstractArray{tp, N}, P::PARAMS)

mutating version of `detectAnomalies()`. Directly writes the output into `P`.
"""
function detectAnomalies!(data::AbstractArray{tp, 2}, P::PARAMS) where {tp}
  copyto!(P.data, data)
  allalgorithms = ["KDE", "REC", "KNN_Delta", "KNN_Gamma", "T2", "SVDD", "KNFST"]
  @assert any(ispartof(P.algorithms, allalgorithms))

  if(any(ispartof(P.algorithms, ["KDE", "REC", "KNN_Delta", "KNN_Gamma"])))   # D based
    dist_matrix!(P.D, P.data, dist = P.dist, Q = P.Q)
    if(any(ispartof(P.algorithms, ["KDE"])))
      kernel_matrix!(P.K, P.D[1], P.K_sigma)
      KDE!(P.KDE, P.K)
    end

     if(any(ispartof(P.algorithms, ["REC"])))
      REC!(P.REC, P.D[1], P.REC_varepsilon, P.temp_excl)
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
    if(any(ispartof(P.algorithms, ["SVDD"])))
      (predicted_labels, decision_values) = SVDD_predict(P.SVDD_model, P.D_test[1])
      P.SVDD = broadcast(*, decision_values, -1)
    end
  end

  if(P.quantiles)
    if(any(ispartof(P.algorithms, ["T2"])))  P.T2_quantiles = get_quantile_scores(P.T2[1]) end
    if(any(ispartof(P.algorithms, ["REC"]))) P.REC_quantiles = get_quantile_scores(P.REC) end
    if(any(ispartof(P.algorithms, ["KDE"]))) P.KDE_quantiles = get_quantile_scores(P.KDE) end
    if(any(ispartof(P.algorithms, ["SVDD"]))) P.SVDD_quantiles = squeeze(get_quantile_scores(P.SVDD), 1) end
    if(any(ispartof(P.algorithms, ["KNFST"]))) P.KNFST_quantiles = get_quantile_scores(P.KNFST[1]) end
    if(any(ispartof(P.algorithms, ["KNN_Gamma"]))) P.KNN_Gamma_quantiles = get_quantile_scores(P.KNN_Gamma) end
    if(any(ispartof(P.algorithms, ["KNN_Delta"]))) P.KNN_Delta_quantiles = get_quantile_scores(P.KNN_Delta[1]) end
  end

  if(P.ensemble_method != "None")
    L = length(P.algorithms)
    if(L > 4 || L < 2) print("compute_ensemble() does currently only support 2-4 detection algorihms. You selected $L. \n") end
    if(!P.quantiles) print("Warning: P.quantiles should be true for computing ensembles out of comparable scores, but is false")
      @assert !any(ispartof(P.algorithms, ["KNN_Delta", "SVDD", "KNFST", "T2"]))
      if(L == 2) P.ensemble = compute_ensemble(getfield(P, Meta.parse(P.algorithms[1])), getfield(P, Meta.parse(P.algorithms[2])), ensemble = P.ensemble_method) end
      if(L == 3) P.ensemble = compute_ensemble(getfield(P, Meta.parse(P.algorithms[1])), getfield(P, Meta.parse(P.algorithms[2])), getfield(P, Meta.parse(P.algorithms[3])), ensemble = P.ensemble_method) end
      if(L == 4) P.ensemble = compute_ensemble(getfield(P, Meta.parse(P.algorithms[1])), getfield(P, Meta.parse(P.algorithms[2])), getfield(P, Meta.parse(P.algorithms[3])), getfield(P, Meta.parse(P.algorithms[4])), ensemble = P.ensemble_method) end
    else
      if(L == 2) P.ensemble = compute_ensemble(getfield(P, Meta.parse(string(P.algorithms[1], "_quantiles"))), getfield(P, Meta.parse(string(P.algorithms[2], "_quantiles"))), ensemble = P.ensemble_method) end
      if(L == 3) P.ensemble = compute_ensemble(getfield(P, Meta.parse(string(P.algorithms[1], "_quantiles"))), getfield(P, Meta.parse(string(P.algorithms[2], "_quantiles"))), getfield(P, Meta.parse(string(P.algorithms[3], "_quantiles"))), ensemble = P.ensemble_method) end
      if(L == 4) P.ensemble = compute_ensemble(getfield(P, Meta.parse(string(P.algorithms[1], "_quantiles"))), getfield(P, Meta.parse(string(P.algorithms[2], "_quantiles"))), getfield(P, Meta.parse(string(P.algorithms[3], "_quantiles"))), getfield(P, Meta.parse(string(P.algorithms[4], "_quantiles"))), ensemble = P.ensemble_method) end
    end
  end

  return(P)
end


"""
    detectAnomalies(data::AbstractArray{tp, N}, P::PARAMS) where {tp, N}
    detectAnomalies(data::AbstractArray{tp, N}, algorithms::Array{String,1} = ["REC", "KDE"]; mean = 0) where {tp, N}

detect anomalies, given some Parameter object `P` of type PARAMS. Train the Parameters `P` with `getParameters()` beforehand on some training data. See `getParameters()`.
Without training `P` beforehand, it is also possible to use `detectAnomalies(data, algorithms)` given some algorithms (except SVDD, KNFST).
Some default parameters are used in this case to initialize `P` internally.

# Examples
```
julia> training_data = randn(100, 2); testing_data = randn(100, 2);
julia> # compute the anoamly scores of the algorithms "REC", "KDE", "T2" and "KNN_Gamma", their quantiles and return their ensemble scores
julia> P = getParameters(["REC", "KDE", "T2", "KNN_Gamma"], training_data, quantiles = true, ensemble_method = "mean");
julia> detectAnomalies(testing_data, P)
```
"""
function detectAnomalies(data::AbstractArray{tp, 2}, P::PARAMS) where {tp}
  init_detectAnomalies(data, P)
  detectAnomalies!(data, P)
  return(return_detectAnomalies(P))
end


function return_scores(i, P::PARAMS)
  if(isa(getfield(P, Meta.parse(P.algorithms[i])), Tuple))
    return(getfield(P, Meta.parse(P.algorithms[i]))[1])
  else
    return(getfield(P, Meta.parse(P.algorithms[i])))
  end
end

function return_quantile_scores(i, P::PARAMS)
    return(getfield(P, Meta.parse(string(P.algorithms[i], "_quantiles"))))
end

function return_detectAnomalies(P::PARAMS)
  L = length(P.algorithms)
  if(any(ispartof([P.ensemble_method], ["mean","max","min", "median"]))) return(P.ensemble)
  elseif(L == 1 && !P.quantiles)  return(return_scores(1,P))
  elseif(!P.quantiles && L > 1) return(ntuple(i->return_scores(i,P), L))
  elseif(L > 1 && P.quantiles) return(ntuple(i->return_quantile_scores(i,P), L))
  elseif(L == 1 && P.quantiles) return(return_quantile_scores(1,P))
  end
end


function detectAnomalies(data::AbstractArray{tp, N}, algorithms::Array{String,1} = ["REC", "KDE"]; mean = 0, dist = "Euclidean") where {tp, N}
  @assert !any(ispartof(algorithms, ["SVDD", "KNFST"]))
   Q = cov(reshape_dims_except_N(data))
   if(mean == 0) meanvec = zeros(Float64, 1, size(data, N))
   else  meanvec = mean(data, dims = 1) end
   D = dist_matrix(data; dist = dist)
   sigma =  median(unsafe_wrap(Array, pointer(D), length(D)))
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
                    , [NaN], [NaN], [NaN], [NaN], [NaN], [NaN], [NaN] # quantiles
                      , data # data
                      , 0, "None", NaN, false
                         )
  init_detectAnomalies(data, P)
  detectAnomalies!(data, P)
  return(return_detectAnomalies(P))
end
