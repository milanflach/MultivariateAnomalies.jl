using MultivariateStats


"""
    sMSC(datacube, cycle_length)

subtract the median seasonal cycle from the datacube given the length of year `cycle_length`.

# Examples

```jldoctest
julia> dc = hcat(rand(193) + 2* sin(0:pi/24:8*pi), rand(193) + 2* sin(0:pi/24:8*pi))
julia> sMSC_dc = sMSC(dc, 48)
```
"""

function sMSC(datacube, cycle_length)
    x_out = similar(datacube)
    removeMSC!(datacube, x_out, cycle_length, 1)
    return(x_out)
end

# fast version of fabian to remove the mean seasonal cycle
function removeMSC!{T,ndim}(xin::Array{T,ndim},xout::Array{T,ndim},NpY::Integer,itimedim::Integer;imscstart::Int=1)
   # Determine length of reshape dimensions
   s=size(xin)
   l1=itimedim==1 ? 1 : prod(s[1:(itimedim-1)])
   l2=itimedim==ndim ? 1 : prod(s[(itimedim+1):end])
   ltime=s[itimedim]
   #Reshape the cube to squeeze unimportant variables
   xin2=reshape(xin,l1,ltime,l2);
   xout2=reshape(xout,l1,ltime,l2);
   msc=zeros(Float64,NpY); # This is the array where the temp msc is stored
   nmsc=zeros(Int,NpY); # This is for counting how many values were added
   #Start loop through all other variables
   for i2=1:l2,i1=1:l1
       imsc=imscstart
       fill!(msc,zero(Float64))
       fill!(nmsc,zero(Int))
       for itime=1:ltime
           curval=xin2[i1,itime,i2]
           if !isnan(curval)
               msc[imsc]  += curval
               nmsc[imsc] += 1
           end
           imsc =imsc==NpY ? 1 : imsc+1 # Increase msc time step counter
       end
       for i in 1:NpY msc[i] = nmsc[i] > 0 ? msc[i]/nmsc[i] : NaN end # Get MSC by dividing by number of points
       imsc=imscstart
       for i in 1:ltime
           xout2[i1,i,i2] = xin2[i1,i,i2]-msc[imsc]
           imsc =imsc==NpY ? 1 : imsc+1 # Increase msc time step counter
       end
   end
   #Copy data if necessary
   pointer(xout2) != pointer(xout) && copy!(xout,xout2)
  return(xout)
end

"""
    globalPCA{tp, N}(datacube::Array{tp, N}, expl_var::Float64 = 0.95)

return an orthogonal subset of the variables, i.e. the last dimension of the datacube.
A Principal Component Analysis is performed on the entire datacube, explaining at least `expl_var` of the variance.
"""

function globalPCA{tp, N}(datacube::AbstractArray{tp, N}, expl_var::Float64 = 0.95)
    if(expl_var > 1.0 || expl_var < 0.0) print("Stop: expl_var has to be within range(0.0, 1.0) but is $expl_var")
    else
    newsize = zeros(Int64, length(size(datacube)))
    dims = 1
    for i = 1:(N-1) # multiply dimensions except the last one and save as dims
        dims = size(datacube, i) * dims
        newsize[i] = size(datacube, i)
    end
    X = reshape(datacube, dims, size(datacube, N)) # reshape according to dims
    X = permutedims(X, [2,1]) # permute as PCA expectes other input format
    M = fit(PCA, X, pratio = 0.999, method = :svd)
    Y = transform(M, X)
    Y = permutedims(Y, [2,1])
    num_expl_comp = findfirst(cumsum(principalvars(M))/tprincipalvar(M) .>= expl_var)
    Y = Y[:,1:num_expl_comp]
    newsize[N] = num_expl_comp
    Xout = similar(datacube)
    Xout = Xout[:,:,:,1:num_expl_comp]
    Xout[:] = Y[:]
    Xout
    end
    return(Xout, principalvars(M)/tprincipalvar(M), cumsum(principalvars(M))/tprincipalvar(M), num_expl_comp)
end

"""
    globalICA(datacube::Array{tp, 4}, mode = "expl_var"; expl_var::Float64 = 0.95, num_comp::Int = 3)

perform an Independent Component Analysis on the entire 4-dimensional datacube either by (`mode = "num_comp"`) returning num_comp number of independent components
or (`mode = "expl_var"`) returning the number of components which is necessary to explain expl_var of the variance, when doing a Prinicpal Component Analysis before.

"""

# ICA with a specific explained percentage of variance out of PCA of a number of components:
function globalICA{tp, N}(datacube::AbstractArray{tp, N}, mode = "expl_var"; expl_var::Float64 = 0.95, num_comp::Int = 3)
    newsize = zeros(Int64, length(size(datacube)))
    dims = 1
    for i = 1:(N-1) # multiply dimensions except the last one and save as dims
        dims = size(datacube, i) * dims
        newsize[i] = size(datacube, i)
    end
    X = reshape(datacube, dims, size(datacube, N)) # reshape according to dims
    X = permutedims(X, [2,1]) # permute as PCA expectes other input format
    if(mode != "expl_var" && mode != "num_comp") print("Stop: mode has to be either 'expl_var' or 'num_comp' but is $mode")
        else
    if (mode == "expl_var")
      if(expl_var > 1.0 || expl_var < 0.0) print("Stop: expl_var has to be within range(0.0, 1.0) but is $expl_var")
        else
        # explaining variance, estimate number of components to a certain variance with PCA
        M = fit(PCA, X, pratio = 0.99, method = :svd)
        num_comp = findfirst(cumsum(principalvars(M))/tprincipalvar(M) .>= expl_var)
      end
    end
    if(num_comp > size(datacube, N)) print("Stop: num_comp has to be smaller than size(datacube, N) = $(size(datacube, N)) but is $num_comp")
      else
        M = fit(ICA, X, num_comp, do_whiten = true, mean = 0.0) # without whitening --> get orthonormal vectors before
        Y = transform(M, X)
        Y = permutedims(Y, [2,1])
        newsize[N] = num_comp
        Xout = similar(datacube)
        Xout = Xout[:,:,:,1:num_comp]
        Xout[:] = Y[:]
        Xout
    end
    return(Xout, principalvars(M)/tprincipalvar(M), cumsum(principalvars(M))/tprincipalvar(M), num_comp)
    end
end

"""
    TDE{tp}(datacube::Array{tp, 4}, ΔT::Integer, DIM::Int = 3)
    TDE{tp}(datacube::Array{tp, 3}, ΔT::Integer, DIM::Int = 3)

returns an embedded datacube by concatenating lagged versions of the 2-, 3- or 4-dimensional datacube with `ΔT` time steps in the past up to dimension `DIM` (presetting: `DIM = 3`)

```jldoctest
julia> dc = randn(50,3)
julia> TDE(dc, 3, 2)
```
"""

function TDE{tp}(datacube::Array{tp, 4}, ΔT::Integer, DIM::Int = 3)
    start = ((DIM-1)*ΔT+1)
    embedded_datacube = zeros(Float64, (size(datacube, 1) - start +1, size(datacube, 2), size(datacube, 3), size(datacube, 4) * DIM))
    for dim = 1:DIM
    embedded_datacube[:,:,:,((dim-1)*size(datacube, 4)+1):(dim*size(datacube, 4))] =
        datacube[(start - ((dim-1) * ΔT)) : (size(datacube, 1) - ((dim-1) * ΔT)),:,:,:]
    end
    return(embedded_datacube)
end

function TDE{tp}(datacube::Array{tp, 3}, ΔT::Integer, DIM::Int = 3)
    start = ((DIM-1)*ΔT+1)
    embedded_datacube = zeros(Float64, (size(datacube, 1) - start +1, size(datacube, 2), size(datacube, 3) * DIM))
    for dim = 1:DIM
    embedded_datacube[:,:,((dim-1)*size(datacube, 3)+1):(dim*size(datacube, 3))] =
        datacube[(start - ((dim-1) * ΔT)) : (size(datacube, 1) - ((dim-1) * ΔT)),:,:]
    end
    return(embedded_datacube)
end

function TDE{tp}(datacube::Array{tp, 2}, ΔT::Integer, DIM::Int = 3)
    start = ((DIM-1)*ΔT+1)
    embedded_datacube = zeros(Float64, (size(datacube, 1) - start +1,  size(datacube, 2) * DIM))
    for dim = 1:DIM
    embedded_datacube[:,((dim-1)*size(datacube, 2)+1):(dim*size(datacube, 2))] =
        datacube[(start - ((dim-1) * ΔT)) : (size(datacube, 1) - ((dim-1) * ΔT)),:]
    end
    return(embedded_datacube)
end

"""
    mw_VAR{tp,N}(datacube::Array{tp,N}, windowsize::Int = 10)

compute the variance in a moving window along the first dimension of the datacube (presetting: `windowsize = 10`).
Accepts N dimensional datacubes.

```jldoctest
julia> dc = randn(50,3,3,3)
julia> mw_VAR(dc, 15)
```
"""

# accepts N- dimensional datacube as input
# function mw_VAR{tp,N}(datacube::Array{tp,N}, windowsize::Int = 10)
#   out = fill(NaN, size(datacube))
#   var(datacube, 1)
#         for t = 1:(size(datacube, 1)-windowsize)
#           #print("i = $i, t = $t \n")
#           copy!(sub(out, Int(t + round(windowsize * 0.5, 0)),:,:), var(sub(datacube,t:(t+windowsize-1),:, :), 1))
#         end
#   return(out)
# end

function mw_VAR{tp,N}(datacube::Array{tp,N}, windowsize::Int = 10)
  out = fill(NaN, size(datacube))
  datacube0mean = datacube .- mean(datacube, 1)
  T = size(datacube, 1)
  for beg = 1:T:length(datacube)
    inner_mw_VAR!(out, datacube0mean, windowsize, beg, T)
  end
  out = out ./ windowsize
  return(out)
end

function inner_mw_VAR!(out, datacube, windowsize, beg, T) # mean already subtracted
  out_beg = Int(floor(windowsize * 0.5)) + beg - 1
  # init x
  x = sum(pointer_to_array(pointer(datacube, beg), windowsize).^2)
  out[out_beg] = x
  @inbounds for i = 1:(T-windowsize +1)
    x = x - datacube[beg + i - 1]^2 + datacube[beg + i - 1 + windowsize - 1]^2
    out[out_beg + i] = x
  end
  return(out)
end


"""
    mw_COR{tp}(datacube::Array{tp, 4}, windowsize::Int = 10)

compute the correlation in a moving window along the first dimension of the datacube (presetting: `windowsize = 10`).
Accepts 4-dimensional datacubes.

"""

function mw_COR{tp}(datacube::Array{tp, 4}, windowsize::Int = 10)
  comb = collect(combinations(1:size(datacube, 4), 2))
  out = zeros(Float64, size(datacube, 1), size(datacube, 2), size(datacube, 3), size(comb, 1))
  x = zeros(Float64, windowsize, 2)
  for icomb = 1:size(comb, 1)
    for lon = 1:size(datacube, 3)
      for lat = 1:size(datacube, 2)
        for t = 1:(size(datacube, 1)-windowsize)
          x = copy!(x, datacube[t:(t+windowsize-1),lat,lon,comb[icomb]])
          copy!(sub(out, t + Int(round(windowsize * 0.5, 0)),lat,lon,icomb)
                , cor(sub(x,:,1), sub(x,:,2)))
        end
      end
    end
  end
  return(comb, out)
end

function innerst_ewma!{tp, N}(z::AbstractArray{tp, N}, dat::AbstractArray{tp, N}, beg::Int, len::Int, lambda::Float64 = 0.15)
    z[beg] = dat[beg]*lambda
    one_m_lambda = 1.0-lambda
    for i = (beg+1):(beg+len-1)
            z[i] = dat[i] * lambda + z[i-1] * one_m_lambda
    end
  return(z)
end

"""
    EWMA(dat,  λ)

Compute the exponential weighted moving average (EWMA) with the weighting parameter `λ` between 0 (full weighting) and 1 (no weighting) in the first dimension of `dat`.
Supports N-dimensional Arrays.

Lowry, C. A., & Woodall, W. H. (1992). A Multivariate Exponentially Weighted Moving Average Control Chart. Technometrics, 34, 46–53.

```jldoctest
julia> dc = rand(100,3,2)
julia> ewma_dc = EWMA(dc, 0.1)
```
"""

function EWMA{tp, N}(dat::AbstractArray{tp, N}, lambda::Float64 = 0.15)
  Z = similar(dat)
  T = size(dat, 1)
  LENGTH = length(dat)
  for istart = 1:T:LENGTH
    innerst_ewma!(Z, dat, istart, T, lambda)
  end
  return(Z)
end


"""
    EWMA!(Z, dat,  λ)

use a preallocated output Z. `Z = similar(dat)` or `dat = dat` for overwriting itself.

# Examples

```jldoctest
julia> dc = rand(100,3,2)
julia> EWMA!(dc, dc, 0.1)
```
"""


function EWMA!{tp, N}(Z::AbstractArray{tp, N}, dat::AbstractArray{tp, N}, lambda::Float64 = 0.15)
  T = size(dat, 1)
  LENGTH = length(dat)
  for istart = 1:T:LENGTH
    innerst_ewma!(Z, dat, istart, T, lambda)
  end
  return(Z)
end

"""
    init_MedianCycle(dat::Array{tp}, cycle_length::Int = 46)
    init_MedianCycle(temporal_length::Int[, cycle_length::Int = 46])

initialises an init_MC object to be used as input for `get_MedianCycle!()`. Input is either some sample data or the temporal lenght of the expected input vector
and the length of the annual cycle (presetting: `cycle_length = 46`)
"""

function init_MedianCycle{tp}(dat::Array{tp}, cycle_length::Int = 46)
  complete_years = Int(floor(size(dat, 1)/cycle_length))
  cycle_dat = zeros(tp, cycle_length, complete_years)
  cycle_medians = zeros(tp, cycle_length)
  init_MC = (cycle_length, complete_years, cycle_medians, cycle_dat)
  return(init_MC)
end

function init_MedianCycle(temporal_length::Int, cycle_length::Int = 46)
  complete_years = Int(floor(temporal_length/cycle_length))
  cycle_dat = zeros(tp, cycle_length, complete_years)
  cycle_medians = zeros(tp, cycle_length)
  init_MC = (cycle_length, complete_years, cycle_medians, cycle_dat)
  return(init_MC)
end

"""
    get_MedianCycle!(init_MC, dat::Array{tp,1})

Memory efficient version of `get_MedianCycle()`, returning the median cycle in `init_MC[3]`. The `init_MC` object should be created with `init_MedianCycle`.
Can deal with some NaN values.

# Examples

```jldoctest
julia> dat = rand(193) + 2* sin(0:pi/24:8*pi)
julia> dat[100] = NaN
julia> init_MC = init_MedianCycle(dat, 48)
julia> get_MedianCycle!(init_MC, dat)
julia> init_MC[3]
```
"""

function get_MedianCycle!{tp}(init_MC::Tuple{Int64,Int64,Array{tp,1},Array{tp,2}}, dat::Array{tp,1})
  copy!(init_MC[4], sub(dat, 1:(init_MC[2] * init_MC[1])))
  for i = 1:init_MC[1]
    if(all(!isnan(sub(init_MC[4], i, :))))
      copy!(sub(init_MC[3], i), median(sub(init_MC[4], i, :)))
    elseif(all(isnan(sub(init_MC[4], i, :))))
        copy!(sub(init_MC[3], i), NaN)
    else
      copy!(sub(init_MC[3], i), median(sub(init_MC[4], i, squeeze(!isnan(sub(init_MC[4], i,:)), 1))))
    end
  end
  return(init_MC[3])
end

"""
    get_MedianCycle(dat::Array{tp,1}, cycle_length::Int = 46)

returns the median annual cycle of a one dimensional data array, given the length of the annual cycle (presetting: `cycle_length = 46`).
Can deal with some NaN values.

# Examples

```jldoctest
julia> dat = rand(193) + 2* sin(0:pi/24:8*pi)
julia> dat[100] = NaN
julia> cycles = get_MedianCycle(dat, 48)
```
"""

function get_MedianCycle{tp}(dat::Array{tp,1}, cycle_length::Int = 46)
  init_MC = init_MedianCycle(dat, cycle_length)
  get_MedianCycle!(init_MC, dat)
  return(init_MC[3])
end


"""
    get_MedianCycles(datacube, cycle_length::Int = 46)

returns the median annual cycle of a datacube, given the length of the annual cycle (presetting: `cycle_length = 46`).
The datacube can be 2, 3, 4-dimensional, time is stored along the first dimension.

# Examples

```jldoctest
julia> dc = hcat(rand(193) + 2* sin(0:pi/24:8*pi), rand(193) + 2* sin(0:pi/24:8*pi))
julia> cycles = get_MedianCycles(dc, 48)
```
"""

function get_MedianCycles{tp}(datacube::Array{tp,4}, cycle_length::Int = 46)
dat = zeros(Float64, size(datacube, 1));
init_MC = init_MedianCycle(dat, cycle_length);
med_cycles_out = zeros(Float64, cycle_length, size(datacube, 2), size(datacube, 3), size(datacube, 4));
# loop
  for var = 1:size(datacube, 4)
    for lon = 1:size(datacube, 3)
      for lat = 1:size(datacube, 2)
        copy!(dat, sub(datacube, :,lat,lon,var))
        copy!(sub(med_cycles_out, :, lat, lon, var), get_MedianCycle!(init_MC, dat))
      end
    end
  end
  return(med_cycles_out)
end

function get_MedianCycles{tp}(datacube::Array{tp,3}, cycle_length::Int = 46)
dat = zeros(tp, size(datacube, 1));
init_MC = init_MedianCycle(dat, cycle_length);
med_cycles_out = zeros(tp, cycle_length, size(datacube, 2), size(datacube, 3));
# loop
  for var = 1:size(datacube, 3)
      for lat = 1:size(datacube, 2)
        copy!(dat, sub(datacube, :, lat,var))
        copy!(sub(med_cycles_out, :, lat, var), get_MedianCycle!(init_MC, dat))
      end
  end
  return(med_cycles_out)
end


function get_MedianCycles{tp}(datacube::Array{tp,2}, cycle_length::Int = 46)
dat = zeros(tp, size(datacube, 1));
init_MC = init_MedianCycle(dat, cycle_length);
med_cycles_out = zeros(tp, cycle_length, size(datacube, 2));
# loop
  for var = 1:size(datacube, 2)
        copy!(dat, sub(datacube, :, var))
        copy!(sub(med_cycles_out, :, var), get_MedianCycle!(init_MC, dat))
  end
  return(med_cycles_out)
end



###################################
#end
