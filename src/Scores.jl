"""
    get_quantile_scores(scores, quantiles = 0.0:0.01:1.0)

return the quantiles of the given N dimensional anomaly `scores` cube. `quantiles` (default: `quantiles = 0.0:0.01:1.0`) is a Float range of quantiles.
Any score being greater or equal `quantiles[i]` and beeing smaller than `quantiles[i+1]` is assigned to the respective quantile `quantiles[i]`.

# Examples
```
julia> scores1 = rand(10, 2)
julia> quantile_scores1 = get_quantile_scores(scores1)
```
"""
function get_quantile_scores(scores::AbstractArray{tp,N}, quantiles::StepRangeLen{Float64} = 0.0:0.01:1.0) where {tp,N}
  quantile_scores = zeros(Float64, size(scores))
  get_quantile_scores!(quantile_scores, scores, quantiles)
  return quantile_scores
end


"""
    get_quantile_scores!{tp,N}(quantile_scores::AbstractArray{Float64, N}, scores::AbstractArray{tp,N}, quantiles::StepRangeLen{Float64} = 0.0:0.01:1.0)

return the quantiles of the given N dimensional `scores` array into a preallocated `quantile_scores` array, see `get_quantile_scores()`.
"""
function get_quantile_scores!(quantile_scores::AbstractArray{Float64, N}, scores::AbstractArray{tp,N}, quantiles::StepRangeLen{Float64} = 0.0:0.01:1.0; return_thresholds::Bool = false) where {tp,N}
  LENGTH = length(scores)
  thresholds = quantile(reshape(scores, prod(size(scores))), collect(quantiles))
  n_quants = size(thresholds, 1)
  for j = 1:LENGTH
      if(scores[j] <= thresholds[1])
        quantile_scores[j] = quantiles[1]
      end
      for i = 2:n_quants
        if(scores[j] > thresholds[i-1] && scores[j] <= thresholds[i])
          quantile_scores[j] = quantiles[i]
        end
      end
  end
  if return_thresholds == false
    return quantile_scores
  else
    return thresholds
  end
end


"""
    compute_ensemble(m1_scores, m2_scores[, m3_scores, m4_scores]; ensemble = "mean")

compute the mean (`ensemble = "mean"`), minimum (`ensemble = "min"`), maximum (`ensemble = "max"`) or median (`ensemble = "median"`) of the given anomaly scores.
Supports between 2 and 4 scores input arrays (`m1_scores, ..., m4_scores`). The scores of the different anomaly detection algorithms should be somehow comparable,
e.g., by using `get_quantile_scores()` before.

# Examples
```
julia> using MultivariateAnomalies
julia> scores1 = rand(10, 2)
julia> scores2 = rand(10, 2)
julia> quantile_scores1 = get_quantile_scores(scores1)
julia> quantile_scores2 = get_quantile_scores(scores2)
julia> compute_ensemble(quantile_scores1, quantile_scores2, ensemble = "max")
```
"""
function compute_ensemble(m1_scores::Array{Tp, N}, m2_scores::Array{Tp, N}; ensemble = "mean") where {Tp, N}
    @assert any(ensemble .== ["mean","min","max","median"])

    scores = cat(N+1,m1_scores, m2_scores);

    if(ensemble == "mean")
      ensemble_scores = squeeze(mean(scores, dims = N+1), N+1)
    end

    if(ensemble == "median")
      ensemble_scores = squeeze(median(scores, dims = N+1), N+1)
    end

    if(ensemble == "max")
      ensemble_scores = squeeze(maximum(scores, dims = N+1), N+1)
    end

    if(ensemble == "min")
      ensemble_scores = squeeze(minimum(scores, dims = N+1), N+1)
    end
  return(ensemble_scores)
end

function compute_ensemble(m1_scores::Array{Tp, N}, m2_scores::Array{Tp, N}, m3_scores::Array{Tp, N}; ensemble = "mean") where {Tp, N}
    @assert any(ensemble .== ["mean","min","max","median"])

    scores = cat(N+1,m1_scores, m2_scores, m3_scores);

    if(ensemble == "mean")
      ensemble_scores = squeeze(mean(scores, dims = N+1), N+1)
    end

    if(ensemble == "median")
      ensemble_scores = squeeze(median(scores, dims = N+1), N+1)
    end

    if(ensemble == "max")
      ensemble_scores = squeeze(maximum(scores, dims = N+1), N+1)
    end

    if(ensemble == "min")
      ensemble_scores = squeeze(minimum(scores, dims = N+1), N+1)
    end
  return(ensemble_scores)
end

function compute_ensemble(m1_scores::Array{Tp, N}, m2_scores::Array{Tp, N}, m3_scores::Array{Tp, N}, m4_scores::Array{Tp, N}; ensemble = "mean") where {Tp, N}
    @assert any(ensemble .== ["mean","min","max","median"])

    scores = cat(N+1,m1_scores, m2_scores, m3_scores, m4_scores);

    if(ensemble == "mean")
      ensemble_scores = squeeze(mean(scores, dims = N+1), N+1)
    end

    if(ensemble == "median")
      ensemble_scores = squeeze(median(scores, dims = N+1), N+1)
    end

    if(ensemble == "max")
      ensemble_scores = squeeze(maximum(scores, dims = N+1), N+1)
    end

    if(ensemble == "min")
      ensemble_scores = squeeze(minimum(scores, dims = N+1), N+1)
    end
  return(ensemble_scores)
end


###################################
# end
