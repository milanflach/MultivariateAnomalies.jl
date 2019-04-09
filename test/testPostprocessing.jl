scores = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
events = [0,1,1,1,0,0]

@test auc(scores, events) == 2/3

scores = [1.0,0.5,0.0,0.25,0.75] .- 1.0

@test all(get_quantile_scores(scores) .== scores .+ 1.0)

x = zeros(10, 3)
y = ones(10, 3)
z = fill(2.0, 10, 3)

@test all(compute_ensemble(x,z, ensemble = "mean") .== ones(10,3))
@test all(compute_ensemble(x,y, ensemble = "max") .== ones(10,3))
@test all(compute_ensemble(x,y, ensemble = "min") .== zeros(10,3))


@test all(compute_ensemble(x,y,z, ensemble = "mean") .== ones(10,3))
@test all(compute_ensemble(x,y,z, ensemble = "max") .== fill(2.0, 10, 3))
@test all(compute_ensemble(x,y,z, ensemble = "min") .== zeros(10,3))
@test all(compute_ensemble(x,y,z, ensemble = "median") .== ones(10,3))

@test all(compute_ensemble(x,y,y,z, ensemble = "mean") .== ones(10,3))
@test all(compute_ensemble(x,y,y,z, ensemble = "max") .== fill(2.0, 10, 3))
@test all(compute_ensemble(x,y,y,z, ensemble = "min") .== zeros(10,3))
@test all(compute_ensemble(x,y,y,z, ensemble = "median") .== ones(10,3))
