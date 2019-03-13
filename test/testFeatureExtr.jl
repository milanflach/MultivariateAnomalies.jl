

# sMSC test
dc = hcat(2* sin.(0:pi/24:8*pi), 3* sin.(0:pi/24:8*pi));
sMSC_dc = sMSC(dc, 48);
@test all(abs.(sMSC_dc) .< 10e-10)


# TDE test
x = [0,0,0,5,0,0,5]
@test all(TDE(x, 3, 2) .== hcat([5,0,0,5],[0,0,0,5]))
# Dim 2
@test all(TDE(reshape(x, 7, 1), 3, 2)        .== reshape(hcat([5,0,0,5],[0,0,0,5]), (4,2)))
# Dim 3
@test all(TDE(reshape(x, 7, 1, 1), 3, 2)     .== reshape(hcat([5,0,0,5],[0,0,0,5]), (4,1,2)))
# Dim 4
@test all(TDE(reshape(x, 7, 1, 1, 1), 3, 2)  .== reshape(hcat([5,0,0,5],[0,0,0,5]), (4,1,1,2)))

# mw_AVG
x = [4.0, 1.0, 1.0, 1.0, 4.0]
@test all(round.(Int, mw_AVG(x, 3)[2:4]) .== [2,1,2])

# mw_VAR
x = [3.0, -3.0, 4.0, -4.0, 5.0]
@test all(round.(Int, mw_VAR(x, 2)[2:5] .* 2) .== [20,25,34, 41])

# EWMA
x = [10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0]
@test all(round.(Int, EWMA(x, 0.5) * 4) .== [20, 10, 5, 2, 1, 21, 10])

# map moving window
x = randn(144, 1)
@test all(reshape(EWMA(reshape(x, 48, 3)')', 144, 1) .== mapMovingWindow(EWMA, x,ObsPerYear = 48, windowsize = 1, numoutvars = 1))
@test all(reshape(EWMA(reshape(x, 48, 3)', 0.5)', 144, 1) .== mapMovingWindow(EWMA, x, 0.5,ObsPerYear = 48, windowsize = 1, numoutvars = 1))

# get median cycle
 x = sind.(0:8:719)
 @test all(get_MedianCycle(x, 45) .== x[1:45])
