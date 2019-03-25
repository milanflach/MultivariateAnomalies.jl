scores = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
events = [0,1,1,1,0,0]

@test auc(scores, events) == 2/3

scores = [1.0,0.5,0.0,0.25,0.75] .- 1.0

@test all(get_quantile_scores(scores) .== scores .+ 1.0)
