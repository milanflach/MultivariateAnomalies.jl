

# sMSC test
dc = hcat(2* sin.(0:pi/24:8*pi), 3* sin.(0:pi/24:8*pi))
sMSC_dc = sMSC(dc, 48)
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
