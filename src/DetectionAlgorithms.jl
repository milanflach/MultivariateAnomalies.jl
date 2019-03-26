using LIBSVM

# function input D is a distance matrix, K is a Kernel matrix
# count number of recurrences per time point i
"""
    REC(D::AbstractArray, rec_threshold::Float64, temp_excl::Int = 0)

Count the number of observations (recurrences) which fall into a radius `rec_threshold` of a distance matrix `D`. Exclude steps which are closer than `temp_excl` to be count as recurrences (default: `temp_excl = 5`)

Marwan, N., Carmen Romano, M., Thiel, M., & Kurths, J. (2007). Recurrence plots for the analysis of complex systems. Physics Reports, 438(5-6), 237–329. http://doi.org/10.1016/j.physrep.2006.11.001
"""
function REC(D::AbstractArray, rec_threshold::Float64, temp_excl::Int = 0)
  rec_out = init_REC(D)
  REC!(rec_out, D, rec_threshold, temp_excl)
  return(rec_out)
end

"""
    init_REC(D::Array{Float64, 2})
    init_REC(T::Int)

get object for memory efficient `REC!()` versions. Input can be a distance matrix `D` or the number of timesteps (observations) `T`.
"""
function init_REC(T::Int)
  rec_out = zeros(Float64, T)
end

function init_REC(D::Array{Float64, 2})
  rec_out = zeros(Float64, size(D, 1))
end

"""
    REC!(rec_out::AbstractArray, D::AbstractArray, rec_threshold::Float64, temp_excl::Int = 0)

Memory efficient version of `REC()` for use within a loop. `rec_out` is preallocated output, should be initialised with `init_REC()`.
"""
function REC!(rec_out::AbstractArray, D::AbstractArray, rec_threshold::Float64, temp_excl::Int = 0)
    N = size(D, 1)
    @assert temp_excl < N - 1
    @inbounds for i = 1:N
    rec_out[i] = 0.0
    if(i-temp_excl-1 >= 1)
      for j = 1:(i-temp_excl-1)
        if D[i, j] < rec_threshold && i != j
          rec_out[i] = rec_out[i] + 1
        end
      end
    end
    if(i+temp_excl+1 <= N)
      for j = (i+temp_excl+1):N
        if D[i, j] < rec_threshold && i != j
          rec_out[i] = rec_out[i] + 1
        end
      end
    end
  end
  broadcast!(/, rec_out, rec_out, N)
  broadcast!(-, rec_out, 1.0, rec_out)
  return(rec_out)
end

"""
    KDE(K)

Compute a Kernel Density Estimation (the Parzen sum), given a Kernel matrix `K`.

Parzen, E. (1962). On Estimation of a Probability Density Function and Mode. The Annals of Mathematical Statistics, 33, 1–1065–1076.
"""
function KDE(K::AbstractArray)
    KDEout = init_KDE(size(K, 1))
    KDE!(KDEout, K)
    return(KDEout)
end

"""
    init_KDE(K::Array{Float64, 2})
    init_KDE(T::Int)

Returns `KDE_out` object for usage in `KDE!()`. Use either a Kernel matrix `K` or the number of time steps/observations `T` as argument.
"""
function init_KDE(T::Int)
  KDE_out = zeros(Float64, T)
end

function init_KDE(K::Array{Float64, 2})
  KDE_out = zeros(Float64, size(K, 1))
end

"""
    KDE!(KDE_out, K)

Memory efficient version of `KDE()`. Additionally uses preallocated `KDE_out` object for writing the results. Initialize `KDE_out` with `init_KDE()`.
"""
function KDE!(KDE_out, K::AbstractArray)
  @assert size(K, 1) == size(KDE_out, 1)
  mean!(KDE_out, K)
  broadcast!(-, KDE_out, 1.0, KDE_out)
  return(KDE_out)
end

"""
    init_T2(VAR::Int, T::Int)
    init_T2{tp}(data::AbstractArray{tp,2})

initialize `t2_out` object for `T2!` either with number of variables `VAR` and observations/time steps `T` or with a two dimensional `data` matrix (time * variables)
"""
function init_T2(VAR::Int, T::Int)
  diagS =  zeros(Float64, VAR, VAR);
  Qinv =  zeros(Float64, VAR, VAR);
  data_norm = zeros(Float64, T, VAR);
  cdata = zeros(Float64, T, VAR);
  maha = zeros(Float64, T);
  t2_out = (maha, diagS, cdata, Qinv, data_norm)
  return(t2_out)
end

function init_T2(data::AbstractArray{tp,N}) where {tp, N}
  VAR = size(data, N)
  T = size(data, 1)
  diagS =  zeros(tp, VAR, VAR);
  Qinv =  zeros(tp, VAR, VAR);
  data_norm = zeros(tp, T, VAR);
  cdata = zeros(tp, T, VAR);
  maha = zeros(tp, T);
  t2_out = (maha, diagS, cdata, Qinv, data_norm)
  return(t2_out)
end

"""
    T2!(t2_out, data, Q[, mv])

Memory efficient version of `T2()`, for usage within a loop etc. Initialize the `t2_out` object with `init_T2()`.
`t2_out[1]` contains the squred Mahalanobis distance after computation.
"""
function T2!(t2_out::Tuple{Array{tp,1},Array{tp,2},Array{tp,2},Array{tp,2},Array{tp,2}}
             , data::AbstractArray{tp,2}, Q::AbstractArray{tp,2}, mv = 0) where {tp}
  (maha, diagS, cdata, Qinv, data_norm) = t2_out
  if(mv == 0)
    copyto!(cdata, data)
  elseif(size(mv, 1) == 1)
    copyto!(cdata, data .- mv)
  elseif(size(mv, 1) != 1)
    copyto!(cdata, data .- mv')
  end
  USVt = svd(Q)
  copyto!(view(diagS, diagind(diagS)), (USVt.S .+ 1e-10)  .^ (-0.5))
  transpose!(Qinv, USVt.U * diagS * USVt.Vt)
  copyto!(data_norm, cdata * Qinv)
  broadcast!(*, data_norm, data_norm, data_norm)
  sum!(maha, data_norm)
  return(t2_out[1])
end



# Hotelling's T^2 (Mahalanobis distance to the data mean)
# input is time * var data matrix
"""
    T2{tp}(data::AbstractArray{tp,2}, Q::AbstractArray[, mv])

Compute Hotelling's T2 control chart (the squared Mahalanobis distance to the data's mean vector (`mv`), given the covariance matrix `Q`).
Input data is a two dimensional data matrix (observations * variables).

Lowry, C. A., & Woodall, W. H. (1992). A Multivariate Exponentially Weighted Moving Average Control Chart. Technometrics, 34, 46–53.
"""
function T2(data::AbstractArray{tp,2}, Q::AbstractArray{tp, 2}, mv = 0) where {tp}
  t2_out = init_T2(data)
  T2!(t2_out, data, Q, mv)
  return(t2_out[1])
end

# mean of k nearest neighbor distances
"""
    KNN_Gamma(knn_dists_out)

This function computes the mean distance of the K nearest neighbors given a `knn_dists_out` object from `knn_dists()` as input argument.

Harmeling, S., Dornhege, G., Tax, D., Meinecke, F., & Müller, K.-R. (2006). From outliers to prototypes: Ordering data. Neurocomputing, 69(13-15), 1608–1618. http://doi.org/10.1016/j.neucom.2005.05.015
"""
function KNN_Gamma(knn_dists_out::Tuple{Int64,Array{Int64,1},Array{Float64,1},Array{Int64,2},Array{Float64,2}})
  NNdists = knn_dists_out[5]
  N = size(NNdists,1)
  KNN_Gamma_out = zeros(Float64, N)
  mean!(KNN_Gamma_out, NNdists)
  return(KNN_Gamma_out)
end


#T: number of timesteps in the datacube
"""
    init_KNN_Gamma(T::Int)
    init_KNN_Gamma(knn_dists_out)

initialize a `KNN_Gamma_out` object for `KNN_Gamma!` either with `T`, the number of observations/time steps or with a `knn_dists_out` object.
"""
function init_KNN_Gamma(T::Int)
  KNN_Gamma_out = zeros(Float64, T)
end

function init_KNN_Gamma(knn_dists_out::Tuple{Int64,Array{Int64,1},Array{Float64,1},Array{Int64,2},Array{Float64,2}})
  KNN_Gamma_out = zeros(Float64, size(knn_dists_out[2], 1))
end


"""
    KNN_Gamma!(KNN_Gamma_out, knn_dists_out)

Memory efficient version of `KNN_Gamma`, to be used in a loop. Initialize `KNN_Gamma_out` with `init_KNN_Gamma()`.
"""
function KNN_Gamma!(KNN_Gamma_out, knn_dists_out::Tuple{Int64,Array{Int64,1},Array{Float64,1},Array{Int64,2},Array{Float64,2}})
  NNdists = knn_dists_out[5]
  @assert size(NNdists,1) ==  size(KNN_Gamma_out,1) || error("input size KNN_Gamma_out and NNdists not equal")
  N = size(NNdists,1)
  mean!(KNN_Gamma_out, NNdists)
  return(KNN_Gamma_out)
end


"""
    init_KNN_Delta(T, VAR, k)

return a `KNN_Delta_out` object to be used for `KNN_Delta!`. Input: time steps/observations `T`, variables `VAR`, number of K nearest neighbors `k`.
"""
function init_KNN_Delta(T::Int, VAR::Int, k::Int)
  r = Array{Float64}(undef, T)
  x_i = Array{Float64}(undef, 1, VAR)
  d_x = Array{Float64}(undef, k, VAR)
  KNN_Delta_out = (r, x_i, d_x)
  return(KNN_Delta_out)
end

function init_KNN_Delta(knn_dists_out::Tuple{Int64,Array{Int64,1},Array{Float64,1},Array{Int64,2},Array{Float64,2}}, VAR::Int)
  T = size(knn_dists_out[2], 1)
  K = knn_dists_out[1]
  r = Array{Float64}(undef, T)
  x_i = Array{Float64}(undef, 1, VAR)
  d_x = Array{Float64}(undef, K, VAR)
  KNN_Delta_out = (r, x_i, d_x)
  return(KNN_Delta_out)
end

"""
    KNN_Delta!(KNN_Delta_out, knn_dists_out, data)

Memory Efficient Version of `KNN_Delta()`. `KNN_Delta_out[1]` is the vector difference of the k-nearest neighbors.
"""
function KNN_Delta!(KNN_Delta_out::Tuple{Array{Float64,1},Array{Float64,2},Array{Float64,2}}
                    , knn_dists_out::Tuple{Int64,Array{Int64,1},Array{Float64,1},Array{Int64,2},Array{Float64,2}}
                    , data::AbstractArray{Float64,2})
    (r, x_i, d_x) = KNN_Delta_out
    indices = knn_dists_out[4]
    K = size(indices, 2)
    T = size(data,1) #dimensions
    VAR = size(data, 2)
    @assert size(x_i, 2) == size(d_x, 2) == VAR || error("size of d_X and x_i not size of size(datacube, 4)")
    @assert size(r, 1) == T || error("size of r not of size(datacube ,1)")
    dists = 0.0
    for i = 1:T
        dists = 0.0
        inds=indices[i,:]
        for k=1:length(inds), j=1:size(data,2)
          d_x[k,j] = data[i,j] - data[inds[k],j]
        end
        sum!(x_i, d_x)
        for j = 1:VAR
            dists += (x_i[1,j] / K)^2
        end
        r[i] = sqrt.(dists)
    end
    return(r)
end

"""
    KNN_Delta(knn_dists_out, data)

Compute Delta as vector difference of the k-nearest neighbors. Arguments are a `knn_dists()` object (`knn_dists_out`) and a `data` matrix (observations * variables)

Harmeling, S., Dornhege, G., Tax, D., Meinecke, F., & Müller, K.-R. (2006). From outliers to prototypes: Ordering data. Neurocomputing, 69(13-15), 1608–1618. http://doi.org/10.1016/j.neucom.2005.05.015
"""
function KNN_Delta(knn_dists_out::Tuple{Int64,Array{Int64,1},Array{Float64,1},Array{Int64,2},Array{Float64,2}}
                    , data::AbstractArray{Float64,2})
  KNN_Delta_out = init_KNN_Delta(knn_dists_out, size(data, 2))
  KNN_Delta!(KNN_Delta_out, knn_dists_out, data)
  return(KNN_Delta_out[1])
end

"""
    init_UNIV(T::Int, VAR::Int)
    init_UNIV{tp}(data::AbstractArray{tp, 2})

initialize a `univ_out` object to be used in `UNIV!()` either with number of time steps/observations `T` and variables `VAR` or with a `data` matrix observations * variables.
"""
function init_UNIV(T::Int, VAR::Int)
    var_dat = zeros(tp, T)
    varquants = zeros(Float64,T, VAR)
    univ_out = (var_dat, varquants)
    return univ_out
end

function init_UNIV(data::AbstractArray{tp, N}) where {tp,N}
     T = size(data, 1)
     VAR = size(data, N)
     var_dat = zeros(tp, T)
     varquants = zeros(Float64,T, VAR)
     univ_out = (var_dat, varquants)
     return univ_out
end

"""
    UNIV!(univ_out, data)

Memory efficient version of `UNIV()`, input an `univ_out` object from `init_UNIV()` and some `data` matrix observations * variables
"""
function UNIV!(univ_out::Tuple{Array{tp,1},Array{Float64,2}}, data::AbstractArray{tp, N}) where {tp, N}
  (var_dat, varquants) = univ_out
  @assert size(var_dat, 1) == size(varquants, 1) == size(data, 1)
  @assert size(varquants, 2) == size(data, 2)
  for variable = 1:size(data, 2)
    # copy with little allocation
    copyto!(var_dat, view(data, :, variable))
    get_quantile_scores!(view(varquants, :, variable), var_dat)
  end
  maximum!(var_dat, varquants)
  return var_dat
end


"""
    UNIV(data)

order the values in each varaible and return their maximum, i.e. any of the variables in `data` (observations * variables) is above a given quantile,
the highest quantile will be returned.
"""
function UNIV(data::AbstractArray{tp, N}) where {tp, N}
  univ_out = init_UNIV(data)
  return UNIV!(univ_out, data)
end

"""
    SVDD_train(K, nu)

train a one class support vecort machine model (i.e. support vector data description), given a kernel matrix K and and the highest possible percentage of outliers `nu`.
Returns the model object (`svdd_model`). Requires LIBSVM.

Tax, D. M. J., & Duin, R. P. W. (1999). Support vector domain description. Pattern Recognition Letters, 20, 1191–1199.
Schölkopf, B., Williamson, R. C., & Bartlett, P. L. (2000). New Support Vector Algorithms. Neural Computation, 12, 1207–1245.
"""
function SVDD_train(K::AbstractArray, nu::Float64)
# function in LIBSVM.jl for optional parameter settings.
  #svdd_model = svmtrain(fill(1, size(K, 1)), K
  #                  , kernel_type = Int32(4), svm_type  = Int32(2), nu = nu
  #                  , probability_estimates = false);
  svdd_model = svmtrain(K,  svmtype = OneClassSVM, nu = nu, kernel = Kernel.Precomputed);
  return(svdd_model)
end

"""
    SVDD_predict(svdd_model, K)

predict the outlierness of an object given the testing Kernel matrix `K` and the `svdd_model` from SVDD_train(). Requires LIBSVM.

Tax, D. M. J., & Duin, R. P. W. (1999). Support vector domain description. Pattern Recognition Letters, 20, 1191–1199.
Schölkopf, B., Williamson, R. C., & Bartlett, P. L. (2000). New Support Vector Algorithms. Neural Computation, 12, 1207–1245.
"""
function SVDD_predict(svdd_model::LIBSVM.SVM, K::AbstractArray)
    (predicted_labels, decision_values) = svmpredict(svdd_model, K)
end


"""
    KNFST_predict(model, K)

predict the outlierness of some data (represented by the kernel matrix `K`), given some KNFST `model` from `KNFST_train(K)`. Compute `K`with `kernel_matrix()`.

Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler:
"Kernel Null Space Methods for Novelty Detection". Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013.

"""
function KNFST_predict(KNFST_mod, K)
  knfst_out = init_KNFST(size(K,2),KNFST_mod)
  KNFST_predict!(knfst_out, KNFST_mod, K)
  return(knfst_out)
end

"""
    init_KNFST(T, KNFST_mod)

initialize a `KNFST_out`object for the use with `KNFST_predict!`, given `T`, the number of observations and the model output `KNFST_train(K)`.
"""
function init_KNFST(T::Int, KNFST_mod::Tuple{Array{Float64,2},Array{Float64,2}})
  diffs = zeros(Float64, T, size(KNFST_mod[2], 2));
  scores = zeros(Float64, T);
  Ktransposed = zeros(Float64, T, size(KNFST_mod[1],1));
  KNFST_out = (scores, diffs, Ktransposed)
  return(KNFST_out)
end


"""
    KNFST_predict!(KNFST_out, KNFST_mod, K)

predict the outlierness of some data (represented by the kernel matrix `K`), given a `KNFST_out` object (`init_KNFST()`), some KNFST model (`KNFST_mod = KNFST_train(K)`)
and the testing kernel matrix K.

Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler:
"Kernel Null Space Methods for Novelty Detection". Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013.
"""
function KNFST_predict!(scores, diffs, Ktransposed, proj, targetValue, K)
  @assert size(scores, 1) == size(K, 2) == size(diffs, 1)
  @assert size(Ktransposed, 1) == size(K, 2) && size(Ktransposed, 2) == size(K, 1)
  transpose!(Ktransposed, K)
  # projected test samples: Ktransposed * model["proj"]
  mul!(diffs, Ktransposed, proj)
  # differences to the target value:
  broadcast!(-,diffs,diffs, targetValue)
  broadcast!(*,diffs, diffs, diffs)
  sum!(scores, diffs)
  broadcast!(sqrt, scores, scores)
  return(scores)
end


function KNFST_predict!(KNFST_out::Tuple{Array{Float64,1},Array{Float64,2},Array{Float64,2}}
                                     , KNFST_mod::Tuple{Array{Float64,2},Array{Float64,2}}, K::Array{Float64,2})
  (scores, diffs, Ktransposed) = KNFST_out
  (proj, targetValue) = KNFST_mod
  KNFST_predict!(scores, diffs, Ktransposed, proj, targetValue, K)
  return(scores)
end

# Learning method for novelty detection with KNFST according to the work:
#
# Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler:
# "Kernel Null Space Methods for Novelty Detection".
# Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013.
#
# Please cite that paper if you are using this code!
#
# proj = calculateKNFST(K, labels)
#
# calculates projection matrix of KNFST
#
# INPUT:
#   K -- (n x n) kernel matrix containing similarities of n training samples
#   labels -- (n x 1) vector containing (multi-class) labels of the n training samples
#
# OUTPUT:
#   proj -- projection matrix for data points (project x via kx*proj,
#           where kx is row vector containing kernel values of x and
#           training data)
#
#
function calculateKNFST(K, labels)

    classes = unique(labels);

    ### check labels
    if length(unique(labels)) == 1
      error("calculateKNFST.jl: not able to calculate a nullspace from data of a single class using KNFST (input variable 'labels' only contains a single value)");
    end

    ### check kernel matrix
    (n,m) = size(K);
    if n != m
        error("calculateKNFST.jl: kernel matrix must be quadratic");
    end

    ### calculate weights of orthonormal basis in kernel space
    centeredK = copy(K); # because we need original K later on again
    centerKernelMatrix(centeredK);
    (basisvecsValues,basisvecs) = eigen(centeredK);
    basisvecs = basisvecs[:,basisvecsValues .> 1e-12];
    basisvecsValues = basisvecsValues[basisvecsValues .> 1e-12];
    basisvecsValues = Diagonal(1 ./ sqrt.(basisvecsValues));
    basisvecs = basisvecs*basisvecsValues;

    ### calculate transformation T of within class scatter Sw:
    ### T= B'*Sw*B = H*H'  and H = B'*K*(I-L) and L a block matrix
    L = zeros(n,n);
    for i=1:length(classes)

       L[labels .== classes[i], labels .== classes[i]] .= 1.0 / sum(labels .== classes[i]);

    end

    ### need Matrix M with all entries 1/m to modify basisvecs which allows usage of
    ### uncentered kernel values:  (eye(size(M))-M)*basisvecs
    M = ones(m,m)./m;

    ### compute helper matrix H
    H = ((I - M)*basisvecs)' * K * (I - L);

    ### T = H*H' = B'*Sw*B with B=basisvecs
    T = H*H';

    ### calculate weights for null space
    eigenvecs = nullspace(T);

    if size(eigenvecs,2) < 1

      (eigenvals,eigenvecs) = eigen(T);
      (min_val,min_ID) = findmin(eigenvals);
      eigenvecs = eigenvecs[:,min_ID];

    end

    ### calculate null space projection and return it
    proj = ((I - M) * basisvecs) * eigenvecs;

end

#############################################################################################################

function centerKernelMatrix(kernelMatrix)
# centering the data in the feature space only using the (uncentered) Kernel-Matrix
#
# INPUT:
#       kernelMatrix -- uncentered kernel matrix
# OUTPUT:
#       centeredKernelMatrix -- centered kernel matrix

  ### get size of kernelMatrix
  n = size(kernelMatrix, 1);

  ### get mean values of each row/column
  columnMeans = mean(kernelMatrix, dims = 2); ### NOTE: columnMeans = rowMeans because kernelMatrix is symmetric
  matrixMean = mean(columnMeans);

  centeredKernelMatrix = kernelMatrix;

  for k=1:n
    for j=1:n
      centeredKernelMatrix[k,j] -= columnMeans[j];
      centeredKernelMatrix[j,k] -= columnMeans[j];
    end
  end
  #This line will not have any effect
  centeredKernelMatrix .+= matrixMean;

end


"""
    KNFST_train(K)

train a one class novelty KNFST model on a Kernel matrix `K` according to
Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler:
"Kernel Null Space Methods for Novelty Detection". Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013.

# Output
`(proj, targetValue)`
`proj` 	-- projection vector for data points (project x via kx*proj, where kx is row vector containing kernel values of x and training data)
`targetValue` -- value of all training samples in the null space
"""
function KNFST_train(K)

    # get number of training samples
    n = size(K,1);

    # include dot products of training samples and the origin in feature space (these dot products are always zero!)
    K_ext = [K zeros(n,1); zeros(1,n) 0];

    # create one-class labels + a different label for the origin
    labels = push!(ones(n),0);

    # get model parameters
    proj = calculateKNFST(K_ext,labels);
    targetValue = mean(K_ext[labels.==1,:]*proj, dims = 1);
    proj = proj[1:n,:];

    # return both variables
    return proj, targetValue

end

"""
    Dist2Centers(centers::AbstractArray{tp, 2}) where {tp}

Compute the distance to the nearest centers of i.e. a K-means clustering output.
Large Distances to the nearest center are anomalies. data: Observations * Variables.
# Example
`(proj, targetValue)`

"""
function Dist2Centers(data::AbstractArray{tp,2}, centers::AbstractArray{tp,2}) where {tp}
  (minD, minDIdx) = findmin(pairwise(Euclidean(), data', centers'), 2)
  clustassgin = zeros(Int, size(data, 1))
  for i = 1:size(data, 1)
    (u, clustassgin[i]) = ind2sub((size(data, 1), size(centers, 1)), minDIdx[i])
  end
  return minD, clustassgin
end


###################################
# end
