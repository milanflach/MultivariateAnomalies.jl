using MultivariateAnomalies
using Base.Test

# write your own tests here
#####################
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
@test all(round(REC(D, sigma, 0), 1) .== round(1 .- ([1,0,2,1,2] ./ 5), 1))
# KNN Gamma, checked manually
@test all(round(KNN_Gamma(knn_dists_out), 1) .== [1.6, 2.1,1.2,1.6,1.2])
# KNN Delta, approximately
@test all(round(KNN_Delta(knn_dists_out, dat), 1) .== [1.4, 2.1, 0.5, 1.6, 0.5])
# KDE # results show exhibit similar ordering like REC
@test all(sortperm(KDE(K)) .== [5,3,1,4,2])
# Hotelling's T^2
# is also the quared mahalanobis distance to the data's mean
using Distances
@test all(round(T2(dat, Q, mean(dat, 1)),2) .== round(pairwise(SqMahalanobis(Q), dat', mean(dat', 2)), 2))

# SVDD
Ktest = exp(-0.5 * pairwise(Euclidean(), dat[1:4,:]', dat') ./ sigma^2)
@test all(SVDD_predict(svdd_model, Ktest)[1] .== [-1, 1, -1, -1, 1])
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
@test round(P.REC, 3) == round(REC(D, sigma, 0), 3)
@test round(P.KDE, 3) == round(KDE(K), 3)
@test round(P.KNN_Gamma, 3) ==  round(KNN_Gamma(knn_dists_out), 3)
@test round(P.KNN_Delta[1], 3) ==  round(KNN_Delta(knn_dists_out, dat), 3)
@test round(P.T2[1], 3) ==  round(T2(dat, Q, mean(dat, 1)),3)

algorithms = ["KNFST", "SVDD"]
P = getParameters(algorithms, dat[1:4,:])
P.K_sigma = sigma
P.SVDD_model = svdd_model
P.KNFST_model = knfst_model
detectAnomalies(dat, P)
(labels, decvalues) = SVDD_predict(svdd_model, Ktest)
@test P.SVDD.class == labels
@test K[1:4,1:4] == P.D_train[1]
@test Ktest == P.D_test[1]
@test round(P.KNFST[1], 3) == round(KNFST_predict(knfst_model, Ktest)[1], 3)
