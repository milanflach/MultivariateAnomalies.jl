import MultivariateStats
import Combinatorics


"""
    sMSC(datacube, cycle_length)

subtract the median seasonal cycle from the datacube given the length of year `cycle_length`.

# Examples
```
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
function removeMSC!(xin::AbstractArray{Tp,ndim},xout::AbstractArray{Tp,ndim},NpY::Integer,itimedim::Integer;imscstart::Int=1) where {Tp,ndim}
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
   pointer(xout2) != pointer(xout) && copyto!(xout,xout2)
  return(xout)
end

"""
    globalPCA(datacube::Array{tp, N}, expl_var::Float64 = 0.95) where {tp, N}

return an orthogonal subset of the variables, i.e. the last dimension of the datacube.
A Principal Component Analysis is performed on the entire datacube, explaining at least `expl_var` of the variance.
"""
function globalPCA(datacube::AbstractArray{tp, N}, expl_var::Float64 = 0.95) where {tp, N}
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
    M = MultivariateStats.fit(MultivariateStats.PCA, X, pratio = 0.999, method = :svd)
    Y = MultivariateStats.transform(M, X)
    Y = permutedims(Y, [2,1])
    num_expl_comp = findfirst(cumsum(MultivariateStats.principalvars(M))/MultivariateStats.tprincipalvar(M) .>= expl_var)
    newsize[N] = num_expl_comp
    Xout = zeros(tp, ntuple(i -> func(i, newsize), size(newsize, 1)))
    copyto!(Xout, unsafe_wrap(Array, pointer(Y),prod(newsize)))
    end
    return(Xout, MultivariateStats.principalvars(M)/MultivariateStats.tprincipalvar(M),
    cumsum(MultivariateStats.principalvars(M))/MultivariateStats.tprincipalvar(M), num_expl_comp)
end

#small helper
function func(i,x)
        return(x[i])
end

# ICA with a specific explained percentage of variance out of PCA of a number of components:
"""
    globalICA(datacube::Array{tp, 4}, mode = "expl_var"; expl_var::Float64 = 0.95, num_comp::Int = 3)

perform an Independent Component Analysis on the entire 4-dimensional datacube either by (`mode = "num_comp"`) returning num_comp number of independent components
or (`mode = "expl_var"`) returning the number of components which is necessary to explain expl_var of the variance, when doing a Prinicpal Component Analysis before.

"""
function globalICA(datacube::AbstractArray{tp, N}, mode = "expl_var"; expl_var::Float64 = 0.95, num_comp::Int = 3) where {tp, N}
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
        M = MultivariateStats.fit(MultivariateStats.PCA, X, pratio = 0.99, method = :svd)
        num_comp = findfirst(cumsum(MultivariateStats.principalvars(M))/MultivariateStats.tprincipalvar(M) .>= expl_var)
      end
    end
    if(num_comp > size(datacube, N)) print("Stop: num_comp has to be smaller than size(datacube, N) = $(size(datacube, N)) but is $num_comp")
      else
        M = MultivariateStats.fit(MultivariateStats.ICA, X, num_comp, do_whiten = true, mean = 0.0) # without whitening --> get orthonormal vectors before
        Y = MultivariateStats.transform(M, X)
        Y = permutedims(Y, [2,1])
        newsize[N] = num_comp
        Xout = similar(datacube)
        Xout = Xout[:,:,:,1:num_comp]
        Xout[:] = Y[:]
        Xout
    end
    return(Xout, num_comp)
    end
end

"""
    TDE(datacube::Array{tp, 4}, ΔT::Integer, DIM::Int = 3) where {tp}
    TDE(datacube::Array{tp, 3}, ΔT::Integer, DIM::Int = 3) where {tp}

returns an embedded datacube by concatenating lagged versions of the 2-, 3- or 4-dimensional datacube with `ΔT` time steps in the past up to dimension `DIM` (presetting: `DIM = 3`)

# Examples
```
julia> dc = randn(50,3)
julia> TDE(dc, 3, 2)
```
"""
function TDE(datacube::AbstractArray{tp, 4}, ΔT::Integer, DIM::Int = 3) where {tp}
    start = ((DIM-1)*ΔT+1)
    embedded_datacube = zeros(Float64, (size(datacube, 1) - start +1, size(datacube, 2), size(datacube, 3), size(datacube, 4) * DIM))
    for dim = 1:DIM
    embedded_datacube[:,:,:,((dim-1)*size(datacube, 4)+1):(dim*size(datacube, 4))] =
        datacube[(start - ((dim-1) * ΔT)) : (size(datacube, 1) - ((dim-1) * ΔT)),:,:,:]
    end
    return(embedded_datacube)
end

function TDE(datacube::AbstractArray{tp, 3}, ΔT::Integer, DIM::Int = 3) where {tp}
    start = ((DIM-1)*ΔT+1)
    embedded_datacube = zeros(Float64, (size(datacube, 1) - start +1, size(datacube, 2), size(datacube, 3) * DIM))
    for dim = 1:DIM
    embedded_datacube[:,:,((dim-1)*size(datacube, 3)+1):(dim*size(datacube, 3))] =
        datacube[(start - ((dim-1) * ΔT)) : (size(datacube, 1) - ((dim-1) * ΔT)),:,:]
    end
    return(embedded_datacube)
end

function TDE(datacube::AbstractArray{tp, 2}, ΔT::Integer, DIM::Int = 3) where {tp}
    start = ((DIM-1)*ΔT+1)
    embedded_datacube = zeros(Float64, (size(datacube, 1) - start +1,  size(datacube, 2) * DIM))
    for dim = 1:DIM
    embedded_datacube[:,((dim-1)*size(datacube, 2)+1):(dim*size(datacube, 2))] =
        datacube[(start - ((dim-1) * ΔT)) : (size(datacube, 1) - ((dim-1) * ΔT)),:]
    end
    return(embedded_datacube)
end

"""
    mw_VAR(datacube::Array{tp,N}, windowsize::Int = 10) where {tp,N}

compute the variance in a moving window along the first dimension of the datacube (presetting: `windowsize = 10`).
Accepts N dimensional datacubes.

# Examples
```
julia> dc = randn(50,3,3,3)
julia> mw_VAR(dc, 15)
```
"""
function mw_VAR(datacube::AbstractArray{tp,N}, windowsize::Int = 10) where {tp,N}
  out = zeros(tp, size(datacube))
  for k = 1:length(out) out[k] = NaN end
  datacube0mean = datacube .- mean(datacube, 1)
  mw_VAR!(out, datacube0mean, windowsize)
  return(out)
end

"""
    mw_VAR!(out::Array{tp, N}, datacube0mean::Array{tp,N}, windowsize::Int = 10) where {tp,N}

mutating version for `mw_VAR()`. The mean of the input data `datacube0mean` has to be 0. Initialize out properly: `out = datacube0mean` leads to wrong results.

"""
function mw_VAR!(out::AbstractArray{tp, N}, datacube0mean::AbstractArray{tp,N}, windowsize::Int = 10) where {tp,N}
  T = size(datacube0mean, 1)
  for beg = 1:T:length(datacube0mean)
    inner_mw_VAR!(out, datacube0mean, windowsize, beg, T)
  end
   broadcast!(/, out, out, windowsize)
  return(out)
end

"""
    inner_mw_VAR!(out::Array{tp, N}, datacube0mean::Array{tp,N}, windowsize::Int = 10) where {tp,N}

internal function for mw_VAR!()

"""
function inner_mw_VAR!(out, datacube, windowsize, beg, T) # mean already subtracted
  out_beg = Int(floor(windowsize * 0.5)) + beg - 1
  for notavailable =  (beg):(out_beg-1)
    out[notavailable] = NaN
  end
  # init x
  x = sum(unsafe_wrap(Array, pointer(datacube, beg), windowsize).^2)
  out[out_beg] = x
  @inbounds for i = 1:(T-windowsize +1)
    x = x - datacube[beg + i - 1]^2 + datacube[beg + i - 1 + windowsize - 1]^2
    out[out_beg + i] = x
  end
  for notavailable = (out_beg + (T-windowsize +1) + 1):(beg+T-1)
    out[notavailable] = NaN
  end
  return(out)
end

"""
    mw_AVG(datacube::AbstractArray{tp,N}, windowsize::Int = 10) where {tp,N}

compute the average in a moving window along the first dimension of the datacube (presetting: `windowsize = 10`).
Accepts N dimensional datacubes.

# Examples
```
julia> dc = randn(50,3,3,3)
julia> mw_AVG(dc, 15)
```
"""
function mw_AVG(datacube::AbstractArray{tp,N}, windowsize::Int = 10) where {tp,N}
  out = zeros(tp, size(datacube))
  for k = 1:length(out) out[k] = NaN end
  T = size(datacube, 1)
  for beg = 1:T:length(datacube)
    inner_mw_AVG!(out, datacube, windowsize, beg, T)
  end
   broadcast!(/, out, out, windowsize)
  return(out)
end

"""
    mw_AVG!(out::Array{tp, N}, datacube::Array{tp,N}, windowsize::Int = 10) where {tp,N}

internal and mutating version for `mw_AVG()`.

"""
function mw_AVG!(out::AbstractArray{tp,N}, datacube::AbstractArray{tp,N}, windowsize::Int = 10) where {tp,N}
  for k = 1:length(out) out[k] = NaN end
  T = size(datacube, 1)
  for beg = 1:T:length(datacube)
    inner_mw_AVG!(out, datacube, windowsize, beg, T)
  end
   broadcast!(/, out, out, windowsize)
  return(out)
end

function inner_mw_AVG!(out::AbstractArray{tp, N}, datacube::AbstractArray{tp, N}, windowsize::Int, beg::Int, T::Int) where {tp,N}
  out_beg = Int(floor(windowsize * 0.5)) + beg - 1
  for notavailable =  (beg):(out_beg-1)
    out[notavailable] = NaN
  end
  # init x
  x = sum(unsafe_wrap(Array, pointer(datacube, beg), windowsize))
  out[out_beg] = x
  @inbounds for i = 1:(T-windowsize +1)
    x = x - datacube[beg + i - 1] + datacube[beg + i - 1 + windowsize - 1]
    out[out_beg + i] = x
  end
  for notavailable = (out_beg + (T-windowsize +1) + 1):(beg+T-1)
    out[notavailable] = NaN
  end
  return(out)
end



"""
    mw_COR(datacube::Array{tp, 4}, windowsize::Int = 10) where {tp}

compute the correlation in a moving window along the first dimension of the datacube (presetting: `windowsize = 10`).
Accepts 4-dimensional datacubes.

"""
function mw_COR(datacube::AbstractArray{tp, 4}, windowsize::Int = 10) where {tp}
  comb = collect(Combinatorics.combinations(1:size(datacube, 4), 2))
  out = zeros(Float64, size(datacube, 1), size(datacube, 2), size(datacube, 3), size(comb, 1))
  x = zeros(Float64, windowsize, 2)
  for icomb = 1:size(comb, 1)
    for lon = 1:size(datacube, 3)
      for lat = 1:size(datacube, 2)
        for t = 1:(size(datacube, 1)-windowsize)
          x = copyto!(x, datacube[t:(t+windowsize-1),lat,lon,comb[icomb]])
          copyto!(view(out, t + round.(Int, windowsize * 0.5),lat,lon,icomb)
                , cor(view(x,:,1), view(x,:,2)))
        end
      end
    end
  end
  return(comb, out)
end

function innerst_ewma!(z::AbstractArray{tp, N}, dat::AbstractArray{tp, N}, beg::Int, len::Int, lambda::Float64 = 0.15) where {tp, N}
    z[beg] = dat[beg]*lambda
    one_m_lambda = 1.0-lambda
    for i = (beg+1):(beg+len-1)
            z[i] = dat[i] * lambda + z[i-1] * one_m_lambda
    end
  return(z)
end

"""
    EWMA(dat,  λ)

Compute the exponential weighted moving average (EWMA) with the weighting parameter `λ` between 0 (full weighting) and 1 (no weighting) along the first dimension of `dat`.
Supports N-dimensional Arrays.

Lowry, C. A., & Woodall, W. H. (1992). A Multivariate Exponentially Weighted Moving Average Control Chart. Technometrics, 34, 46–53.

# Examples
```
julia> dc = rand(100,3,2)
julia> ewma_dc = EWMA(dc, 0.1)
```
"""
function EWMA(dat::AbstractArray{tp, N}, lambda::Float64 = 0.15) where {tp, N}
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
```
julia> dc = rand(100,3,2)
julia> EWMA!(dc, dc, 0.1)
```
"""
function EWMA!(Z::AbstractArray{tp, N}, dat::AbstractArray{tp, N}, lambda::Float64 = 0.15) where {tp, N}
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
function init_MedianCycle(dat::AbstractArray{tp}, cycle_length::Int = 46) where {tp}
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
```
julia> using MultivariateAnomalies
julia> dat = rand(193) + 2* sin(0:pi/24:8*pi)
julia> dat[100] = NaN
julia> init_MC = init_MedianCycle(dat, 48)
julia> get_MedianCycle!(init_MC, dat)
julia> init_MC[3]
```
"""
function get_MedianCycle!(init_MC::Tuple{Int64,Int64,Array{tp,1},Array{tp,2}}, dat::Array{tp,1}) where {tp}
  copyto!(init_MC[4], view(dat, 1:(init_MC[2] * init_MC[1])))
  for i = 1:init_MC[1]
    if(all(.!isnan.(view(init_MC[4], i, :))))
      copyto!(view(init_MC[3], i), median(view(init_MC[4], i, :)))
    elseif(all(isnan.(view(init_MC[4], i, :))))
        copyto!(view(init_MC[3], i), NaN)
    else
      copyto!(view(init_MC[3], i), median(view(init_MC[4], i, squeeze(!isnan.(view(init_MC[4], i,:)), 1))))
    end
  end
  return(init_MC[3])
end

"""
    get_MedianCycle(dat::Array{tp,1}, cycle_length::Int = 46)

returns the median annual cycle of a one dimensional data array, given the length of the annual cycle (presetting: `cycle_length = 46`).
Can deal with some NaN values.

# Examples
```
julia> using MultivariateAnomalies
julia> dat = rand(193) + 2* sin(0:pi/24:8*pi)
julia> dat[100] = NaN
julia> cycles = get_MedianCycle(dat, 48)
```
"""
function get_MedianCycle(dat::AbstractArray{tp,1}, cycle_length::Int = 46) where {tp}
  init_MC = init_MedianCycle(dat, cycle_length)
  get_MedianCycle!(init_MC, dat)
  return(init_MC[3])
end


"""
    get_MedianCycles(datacube, cycle_length::Int = 46)

returns the median annual cycle of a datacube, given the length of the annual cycle (presetting: `cycle_length = 46`).
The datacube can be 2, 3, 4-dimensional, time is stored along the first dimension.

# Examples
```
julia> using MultivariateAnomalies
julia> dc = hcat(rand(193) + 2* sin(0:pi/24:8*pi), rand(193) + 2* sin(0:pi/24:8*pi))
julia> cycles = get_MedianCycles(dc, 48)
```
"""
function get_MedianCycles(datacube::AbstractArray{tp,4}, cycle_length::Int = 46) where {tp}
dat = zeros(Float64, size(datacube, 1));
init_MC = init_MedianCycle(dat, cycle_length);
med_cycles_out = zeros(Float64, cycle_length, size(datacube, 2), size(datacube, 3), size(datacube, 4));
# loop
  for var = 1:size(datacube, 4)
    for lon = 1:size(datacube, 3)
      for lat = 1:size(datacube, 2)
        copyto!(dat, view(datacube, :,lat,lon,var))
        copyto!(view(med_cycles_out, :, lat, lon, var), get_MedianCycle!(init_MC, dat))
      end
    end
  end
  return(med_cycles_out)
end

function get_MedianCycles(datacube::AbstractArray{tp,3}, cycle_length::Int = 46) where {tp}
dat = zeros(tp, size(datacube, 1));
init_MC = init_MedianCycle(dat, cycle_length);
med_cycles_out = zeros(tp, cycle_length, size(datacube, 2), size(datacube, 3));
# loop
  for var = 1:size(datacube, 3)
      for lat = 1:size(datacube, 2)
        copyto!(dat, view(datacube, :, lat,var))
        copyto!(view(med_cycles_out, :, lat, var), get_MedianCycle!(init_MC, dat))
      end
  end
  return(med_cycles_out)
end


function get_MedianCycles(datacube::AbstractArray{tp,2}, cycle_length::Int = 46) where {tp}
dat = zeros(tp, size(datacube, 1));
init_MC = init_MedianCycle(dat, cycle_length);
med_cycles_out = zeros(tp, cycle_length, size(datacube, 2));
# loop
  for var = 1:size(datacube, 2)
        copyto!(dat, view(datacube, :, var))
        copyto!(view(med_cycles_out, :, var), get_MedianCycle!(init_MC, dat))
  end
  return(med_cycles_out)
end



###################################
#end
