# DistDensity.jl
dat = [[0.0 1.0];[1.0 -1.0];[2.0 2.0];[2.0 3.0];[1.0 1.0]]
D = dist_matrix(dat)
knn_dists_out = knn_dists(D, 2, 0)
@test all(vec(knn_dists_out[4]) .== [5,5,4,3,1,2,1,5,5,3]) || all(vec(knn_dists_out[4]) .== [5,5,4,3,1,3,1,5,5,3])
# parameters
Q = zeros(Float64, 2,2); Q[1,1] = 1.0; Q[2,2] = 1.0
sigma = median(D)
K = kernel_matrix(D, sigma)
#####################
# DetectionAlgorithms.jl
# model
svdd_model = SVDD_train(K[1:4,1:4], 0.2);
knfst_model = KNFST_train(K[1:4,1:4])
# REC, checked manually
@test all(round.(REC(D, sigma, 0), digits = 1) .== round.(1 .- ([1,0,2,1,2] ./ 5), digits = 1))
# KNN Gamma, checked manually
@test all(round.(KNN_Gamma(knn_dists_out), digits = 1) .== [1.6, 2.1,1.2,1.6,1.2])
# KNN Delta, approximately
@test all(round.(KNN_Delta(knn_dists_out, dat), digits = 1) .== [1.4, 2.1, 0.5, 1.6, 0.5])
# KDE # results show exhibit similar ordering like REC
@test all(sortperm(KDE(K)) .== [5,3,1,4,2])
# Hotelling's T^2
# is also the quared mahalanobis distance to the data's mean
using Distances
@test all(round.(T2(dat, Q, mean(dat, dims = 1)), digits = 2) .== round.(pairwise(SqMahalanobis(Q), dat', mean(dat', dims = 2), dims = 2), digits = 2))

# SVDD
Ktest = exp.(-0.5 * pairwise(Euclidean(), dat[1:4,:]', dat') ./ sigma^2)
@test all(SVDD_predict(svdd_model, Ktest)[1] .== [0, 1, 0, 0, 1])
# KNFST, last data point (not seen in training) should differ, i.e. have largest values
@test sortperm(-KNFST_predict(knfst_model, Ktest)[1])[1] == 5

# high level functions
algorithms = ["T2", "REC", "KDE", "KNN_Gamma", "KNN_Delta"]
P = getParameters(algorithms, dat)
P.KNN_k = 2
P.Q = Q
@test P.K_sigma == sigma == P.REC_varepsilon
# detectAnomalies
detectAnomalies(dat, P)
# chekct detectAnomalies for self consistency
@test round.(P.REC, digits = 3) == round.(REC(D, sigma, 0), digits = 3)
@test round.(P.KDE, digits = 3) == round.(KDE(K), digits = 3)
@test round.(P.KNN_Gamma, digits = 3) ==  round.(KNN_Gamma(knn_dists_out), digits = 3)
@test round.(P.KNN_Delta[1], digits = 3) ==  round.(KNN_Delta(knn_dists_out, dat), digits = 3)
@test round.(P.T2[1], digits = 3) ==  round.(T2(dat, Q, mean(dat, dims = 1)), digits = 3)

algorithms = ["KNFST", "SVDD"]
P = getParameters(algorithms, dat[1:4,:])
P.K_sigma = sigma
P.SVDD_model = svdd_model
P.KNFST_model = knfst_model
detectAnomalies(dat, P)
(labels, decvalues) = SVDD_predict(svdd_model, Ktest)
@test round.(P.SVDD, digits = 3) == round.(decvalues, digits = 3) * -1
@test K[1:4,1:4] == P.D_train[1]
@test Ktest == P.D_test[1]
@test round.(P.KNFST[1], digits = 3) == round.(KNFST_predict(knfst_model, Ktest)[1], digits = 3)


data = rand(200,3)
methods = ["REC","KDE"]
@test all(detectAnomalies(data, getParameters(methods, data)) .== detectAnomalies(data, methods))


x = [[1.0,2.0,3.0] [-2.0,4.0,1.0]]
@test all(round.(UNIV(x), digits = 0) .== [0,1,1])

# test online algorithms with the non-online corresponding ones
using StatsBase

out = zeros(100)
x = randn(100, 3)
Q = StatsBase.cov(x)

@test all(round.(KDEonline!(out, x, 1.0), digits = 4) .== round.(KDE(kernel_matrix(dist_matrix(x), 1.0)), digits = 4))

@test all(round.(KDEonline!(out, x, Q, 1.0), digits = 4) .== round.(KDE(kernel_matrix(dist_matrix(x, dist = "Mahalanobis", Q = Q), 1.0)), digits = 4))

@test all(round.(REConline!(out, x, 1.0), digits = 4) .== round.(REC(dist_matrix(x), 1.0), digits = 4))

@test all(round.(REConline!(out, x, Q, 1.0), digits = 4) .== round.(REC(dist_matrix(x, dist = "Mahalanobis", Q = Q), 1.0), digits = 4))

@test all(round.(KNNonline!(out, x, 5), digits = 4) .== round.(KNN_Gamma(knn_dists(dist_matrix(x, dims = 2), 5)), digits = 4))

@test all(round.(KNNonline!(out, x, Q, 5), digits = 4) .== round.(KNN_Gamma(knn_dists(dist_matrix(x, dist = "Mahalanobis",Q = Q, dims = 2), 5)), digits = 4))

s = zeros(1)

SigmaOnline!(s, x)

@test floor.(Int, s)[1] == 2 ||  floor.(Int, s)[1] == 1

SigmaOnline!(s, x, Q)

@test floor.(Int, s)[1] == 2 ||  floor.(Int, s)[1] == 1
