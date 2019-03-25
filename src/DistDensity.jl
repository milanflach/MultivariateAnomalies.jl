using Distances

"""
    init_dist_matrix(data)
    init_dist_matrix(data, training_data)

initialize a `D_out` object for `dist_matrix!()`.
"""
function init_dist_matrix(data::AbstractArray{tp, N}) where {tp, N}
  T = size(data, 1)
  dat = zeros(tp, T, size(data, N))
  tdat = zeros(tp, size(data, N), T)
  D = zeros(tp, T, T)
  D_out = (D, dat, tdat)
  return(D_out)
end

function init_dist_matrix(data::AbstractArray{tp, N}, training_data::AbstractArray{tp, N}) where {tp, N}
  T = size(data, 1)
  Ttrain = size(training_data, 1)
  dat = zeros(tp, T, size(data, N))
  traindat = zeros(tp, Ttrain, size(training_data, N))
  tdat = zeros(tp, size(data, N), T)
  ttraindat = zeros(tp, size(training_data, N), Ttrain)
  D = zeros(tp, Ttrain, T)
  D_out = (D, dat, tdat, traindat, ttraindat)
  return(D_out)
end

"""
    dist_matrix!(D_out, data, ...)

compute the distance matrix of `data`, similar to `dist_matrix()`. `D_out` object has to be preallocated, i.e. with `init_dist_matrix`.

# Examples
```
julia> dc = randn(10,4, 4,3)
julia> D_out = init_dist_matrix(dc)
julia> dist_matrix!(D_out, dc, lat = 2, lon = 2)
julia> D_out[1]
```
"""
function dist_matrix!(D_out::Tuple{Array{tp,2},Array{tp,2},Array{tp,2}}, data::AbstractArray{tp, N}; dist::String = "Euclidean", space::Int = 0, lat::Int = 0, lon::Int = 0, Q = 0, dims = 2) where {tp, N}
  #@assert N == 2 || N == 3 || N  = 4
  (D, dat, tdat) = D_out
  if N == 2 copyto!(dat, data) end
  if N == 3 copyto!(dat, view(data, :, space, :)) end
  if N == 4 copyto!(dat, view(data, :, lat, lon, :))  end
  transpose!(tdat, dat)
  if(dist == "Euclidean")         pairwise!(D, Euclidean(), tdat, dims = dims)
  elseif(dist == "SqEuclidean")   pairwise!(D, SqEuclidean(), tdat, dims = dims)
  elseif(dist == "Chebyshev")     pairwise!(D, Chebyshev(), tdat, dims = dims)
  elseif(dist == "Cityblock")     pairwise!(D, Cityblock(), tdat, dims = dims)
  elseif(dist == "JSDivergence")  pairwise!(D, JSDivergence(), tdat, dims = dims)
  elseif(dist == "Mahalanobis")   pairwise!(D, Mahalanobis(Q), tdat, dims = dims)
  elseif(dist == "SqMahalanobis") pairwise!(D, SqMahalanobis(Q), tdat, dims = dims)
  else print("$dist is not a defined distance metric, has to be one of 'Euclidean', 'SqEuclidean', 'Chebyshev', 'Cityblock' or 'JSDivergence'")
  end
  return(D_out[1])
end

function dist_matrix!(D_out::Tuple{Array{tp,2},Array{tp,2},Array{tp,2},Array{tp,2},Array{tp,2}},
                             data::AbstractArray{tp, N}, training_data::AbstractArray{tp, N}; dist::String = "Euclidean", space::Int = 0, lat::Int = 0, lon::Int = 0, Q = 0, dims = 2) where {tp, N}
  #@assert N == 2 || N == 3 || N  = 4
  (D, dat, tdat, traindat, ttraindat) = D_out
  if N == 2 copyto!(dat, data) end
  if N == 3 copyto!(dat, view(data, :, space, :)) end
  if N == 4 copyto!(dat, view(data, :, lat, lon, :))  end
  if N == 2 copyto!(traindat, training_data) end
  if N == 3 copyto!(traindat, view(training_data, :, space, :)) end
  if N == 4 copyto!(traindat, view(training_data, :, lat, lon, :))  end
  transpose!(tdat, dat)
  transpose!(ttraindat, traindat)
  if(dist == "Euclidean")         pairwise!(D, Euclidean(), ttraindat, tdat, dims = dims)
  elseif(dist == "SqEuclidean")   pairwise!(D, SqEuclidean(), ttraindat, tdat, dims = dims)
  elseif(dist == "Chebyshev")     pairwise!(D, Chebyshev(), ttraindat, tdat, dims = dims)
  elseif(dist == "Cityblock")     pairwise!(D, Cityblock(), ttraindat, tdat, dims = dims)
  elseif(dist == "JSDivergence")  pairwise!(D, JSDivergence(), ttraindat, tdat, dims = dims)
  elseif(dist == "Mahalanobis")   pairwise!(D, Mahalanobis(Q), ttraindat, tdat, dims = dims)
  elseif(dist == "SqMahalanobis") pairwise!(D, SqMahalanobis(Q), ttraindat, tdat, dims = dims)
  else print("$dist is not a defined distance metric, has to be one of 'Euclidean', 'SqEuclidean', 'Chebyshev', 'Cityblock' or 'JSDivergence'")
  end
  return(D_out[1])
end

"""
    dist_matrix(data::AbstractArray{tp, N}; dist::String = "Euclidean", space::Int = 0, lat::Int = 0, lon::Int = 0, Q = 0) where {tp, N}
    dist_matrix(data::AbstractArray{tp, N}, training_data; dist::String = "Euclidean", space::Int = 0, lat::Int = 0, lon::Int = 0, Q = 0) where {tp, N}

compute the distance matrix of `data` or the distance matrix between data and training data i.e. the pairwise distances along the first dimension of data, using the last dimension as variables.
`dist` is a distance metric, currently `Euclidean`(default), `SqEuclidean`, `Chebyshev`, `Cityblock`, `JSDivergence`, `Mahalanobis` and `SqMahalanobis` are supported.
The latter two need a covariance matrix `Q` as input argument.

# Examples
```
julia> dc = randn(10, 4,3)
julia> D = dist_matrix(dc, space = 2)
```
"""
function dist_matrix(data::AbstractArray{tp, N}; dist::String = "Euclidean", space::Int = 0, lat::Int = 0, lon::Int = 0, Q = 0, dims::Int = 2) where {tp, N}
  D_out = init_dist_matrix(data)
  dist_matrix!(D_out, data, dist = dist, space = space, lat = lat, lon = lon ,Q = Q, dims = dims)
  return(D_out[1])
end

function dist_matrix(data::AbstractArray{tp, N}, training_data::AbstractArray{tp, N}; dist::String = "Euclidean", space::Int = 0, lat::Int = 0, lon::Int = 0, Q = 0, dims::Int = 2) where {tp, N}
  D_out = init_dist_matrix(data, training_data)
  dist_matrix!(D_out, data, training_data, dist = dist, space = space, lat = lat, lon = lon ,Q = Q, dims = dims)
  return(D_out[1])
end

"""
    knn_dists(D, k::Int, temp_excl::Int = 0)

returns the k-nearest neighbors of a distance matrix `D`. Excludes `temp_excl` (default: `temp_excl = 0`) distances
from the main diagonal of `D` to be also nearest neighbors.

# Examples
```
julia> dc = randn(20, 4,3)
julia> D = dist_matrix(dc, space = 2)
julia> knn_dists_out = knn_dists(D, 3, 1)
julia> knn_dists_out[5] # distances
julia> knn_dists_out[4] # indices
```
"""
function knn_dists(D::AbstractArray, k::Int, temp_excl::Int = 0)
    T = size(D,1)
    if ((k + temp_excl) > T-1) print("k has to be smaller size(D,1)") end
    knn_dists_out = init_knn_dists(T, k)
    knn_dists!(knn_dists_out, D, temp_excl)
    return(knn_dists_out)
end

"""
    init_knn_dists(T::Int, k::Int)
    init_knn_dists(datacube::AbstractArray, k::Int)

initialize a preallocated `knn_dists_out` object. `k`is the number of nerarest neighbors, `T` the number of time steps (i.e. size of the first dimension) or a multidimensional `datacube`.
"""
function init_knn_dists(T::Int, k::Int)
    ix = zeros(Int64, T)
    v = zeros(Float64, T)
    indices = zeros(Int64, T, k)
    nndists = zeros(Float64, T, k)
    knn_dists_out = (k, ix, v, indices, nndists)
    return(knn_dists_out)
end

function init_knn_dists(datacube::AbstractArray, k::Int)
    T = size(datacube, 1)
    ix = zeros(Int64, T)
    v = zeros(Float64, T)
    indices = zeros(Int64, T, k)
    nndists = zeros(Float64, T, k)
    knn_dists_out = (k, ix, v, indices, nndists)
    return(knn_dists_out)
end

"""
    knn_dists!(knn_dists_out, D, temp_excl::Int = 0)

returns the k-nearest neighbors of a distance matrix `D`. Similar to `knn_dists()`, but uses preallocated input object `knn_dists_out`, initialized with `init_knn_dists()`.
Please note that the number of nearest neighbors `k` is not necessary, as it is already determined by the `knn_dists_out` object.

# Examples
```
julia> dc = randn(20, 4,3)
julia> D = dist_matrix(dc, space = 2)
julia> knn_dists_out = init_knn_dists(dc, 3)
julia> knn_dists!(knn_dists_out, D)
julia> knn_dists_out[5] # distances
julia> knn_dists_out[4] # indices
```
"""
function knn_dists!(knn_dists_out::Tuple{Int64,Array{Int64,1},Array{Float64,1},Array{Int64,2},Array{Float64,2}}, D::AbstractArray, temp_excl::Int = 0)
    (k, ix, v, indices, nndists) = knn_dists_out
    T = size(D,1)
    if ((k + temp_excl) > T-1) print("k has to be smaller size(D,1)") end
    maxD = maximum(D)
    for i = 1:T
        copyto!(v, view(D,:,i))
        for excl = -temp_excl:temp_excl
          if(i+excl > 0 && i+excl <= T)
            v[i+excl]= maxD
          end
        end
        sortperm!(ix, v)
        for j = 1:k
            indices[i,j] = ix[j]
            nndists[i,j] = v[ix[j]]
        end
    end
    return(knn_dists_out)
end

# compute kernel matrix from distance matrix
"""
    kernel_matrix(D::AbstractArray, σ::Float64 = 1.0[, kernel::String = "gauss", dimension::Int64 = 1])

compute a kernel matrix out of distance matrix `D`, given `σ`. Optionally normalized by the `dimension`, if `kernel = "normalized_gauss"`.
compute `D` with `dist_matrix()`.

# Examples
```
julia> dc = randn(20, 4,3)
julia> D = dist_matrix(dc, space = 2)
julia> K = kernel_matrix(D, 2.0)
```
"""
function kernel_matrix(D::AbstractArray, σ::Float64 = 1.0, kernel::String = "gauss", dimension::Int64 = 1)
  K = similar(D)
  kernel_matrix!(K, D, σ, kernel, dimension)
  return K
end

# compute kernel matrix from distance matrix
"""
    kernel_matrix!(K, D::AbstractArray, σ::Float64 = 1.0[, kernel::String = "gauss", dimension::Int64 = 1])

compute a kernel matrix out of distance matrix `D`. Similar to `kernel_matrix()`, but with preallocated Array K (`K = similar(D)`) for output.

# Examples
```
julia> dc = randn(20, 4,3)
julia> D = dist_matrix(dc, space = 2)
julia> kernel_matrix!(D, D, 2.0) # overwrites distance matrix
```
"""
function kernel_matrix!(K::AbstractArray{T,N}, D::AbstractArray{T,N}, σ::Real = 1.0, kernel::String = "gauss", dimension::Int64 = 10) where {T,N}
    #if(size(D, 1) != size(D, 2)) print("D is not a distance matrix with equal dimensions")
    σ = convert(T, σ)
    if(kernel == "normalized_gauss") # k integral gets one
    for i in eachindex(K)
      @inbounds K[i] = exp.(-0.5 * D[i]./(σ*σ))./(σ * (2 *pi).^(dimension/2))#exp.(-0.5 * D[i]./(σ*σ))./((2 *pi*σ*σ).^(dimension/2))
    end
    elseif (kernel == "gauss")
    for i in eachindex(K)
        @inbounds K[i] = exp.(-0.5 * D[i]./(σ*σ))
    end
    end
  return(K)
end


###################################
#end
