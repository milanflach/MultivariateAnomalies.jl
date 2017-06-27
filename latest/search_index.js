var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#MultivariateAnomalies.jl-1",
    "page": "Home",
    "title": "MultivariateAnomalies.jl",
    "category": "section",
    "text": "A julia package for detecting multivariate anomalies.Keywords: Novelty detection, Anomaly Detection, Outlier Detection, Statistical Process ControlPlease cite this package as: Flach, M., Gans, F., Brenning, A., Denzler, J., Reichstein, M., Rodner, E., Bathiany, S., Bodesheim, P., Guanche, Y., Sippel, S., and Mahecha, M. D.: Multivariate Anomaly Detection for Earth Observations: A Comparison of Algorithms and Feature Extraction Techniques, Earth Syst. Dynam. Discuss., in review, 2016. doi:10.5194/esd-2016-51."
},

{
    "location": "index.html#Requirements-1",
    "page": "Home",
    "title": "Requirements",
    "category": "section",
    "text": "Julia 0.5\nJulia packages Distances, MultivariateStats and LIBSVM.\nImportant to get the latest LIBSVM branch via:Pkg.clone(\"https://github.com/milanflach/LIBSVM.jl.git\"); Pkg.checkout(\"LIBSVM\", \"mutating_versions\"); Pkg.build(\"LIBSVM\")"
},

{
    "location": "index.html#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "clone the package: Pkg.clone(\"https://github.com/milanflach/MultivariateAnomalies.jl\")"
},

{
    "location": "index.html#Package-Features-1",
    "page": "Home",
    "title": "Package Features",
    "category": "section",
    "text": "Detect anomalies in your data with easy to use high level functions or individual anomaly detection algorithms\nFeature Extraction: Preprocess your data by extracting relevant features\nSimilarities and Dissimilarities: Compute distance matrices, kernel matrices and k-nearest neighbor objects.\nPostprocessing: Postprocess your anomaly scores, by computing their quantiles or combinations of several algorithms (ensembles).\nAUC: Compute the area under the curve as external evaluation metric of your scores.\nOnline Algorithms: Algorithms tuned for little memory allocation."
},

{
    "location": "index.html#Using-the-Package-1",
    "page": "Home",
    "title": "Using the Package",
    "category": "section",
    "text": "For a quick start it might be useful to start with the high level functions for detecting anomalies. They can be used in highly automized way. "
},

{
    "location": "index.html#Input-Data-1",
    "page": "Home",
    "title": "Input Data",
    "category": "section",
    "text": "MultivariateAnomalies.jl assumes that observations/samples/time steps are stored along the first dimension of the data array (rows of a matrix) with the number of observations T = size(data, 1). Variables/attributes are stored along the last dimension N of the data array (along the columns of a matrix) with the number of variables VAR = size(data, N). The implemented anomaly detection algorithms return anomaly scores indicating which observation(s) of the data are anomalous."
},

{
    "location": "index.html#Authors-1",
    "page": "Home",
    "title": "Authors",
    "category": "section",
    "text": "The package was implemented by Milan Flach and Fabian Gans, Max Planck Institute for Biogeochemistry, Department Biogeochemical Integration, Jena."
},

{
    "location": "index.html#Index-1",
    "page": "Home",
    "title": "Index",
    "category": "section",
    "text": "Pages = [\"man/Preprocessing.md\", \"man/DetectionAlgorithms.md\", \"man/Postprocessing.md\", \"man/AUC.md\", \"man/DistancesDensity.md\", \"man/OnlineAlgorithms.md\"]"
},

{
    "location": "man/HighLevelFunctions.html#",
    "page": "High Level Functions",
    "title": "High Level Functions",
    "category": "page",
    "text": ""
},

{
    "location": "man/HighLevelFunctions.html#High-Level-Anomaly-Detection-Algorithms-1",
    "page": "High Level Functions",
    "title": "High Level Anomaly Detection Algorithms",
    "category": "section",
    "text": "We provide high-level convenience functions for detecting the anomalies. Namely the pair of P = getParameters(algorithms, training_data)  and detectAnomalies(testing_data, P)sets standard choices of the Parameters P and hands the parameters as well as the algorithms choice over to detect the anomalies. Currently supported algorithms include Kernel Density Estimation (algorithms = [\"KDE\"]), Recurrences (\"REC\"), k-Nearest Neighbors algorithms (\"KNN-Gamma\", \"KNN-Delta\"), Hotelling's T^2 (\"T2\"), Support Vector Data Description (\"SVDD\") and Kernel Null Foley Summon Transform (\"KNFST\"). With getParameters() it is also possible to compute output scores of multiple algorithms at once (algorihtms = [\"KDE\", \"T2\"]), quantiles of the output anomaly scores (quantiles = true) and ensembles of the selected algorithms (e.g. ensemble_method = \"mean\"). "
},

{
    "location": "man/HighLevelFunctions.html#MultivariateAnomalies.getParameters",
    "page": "High Level Functions",
    "title": "MultivariateAnomalies.getParameters",
    "category": "Function",
    "text": "getParameters(algorithms::Array{String,1} = [\"REC\", \"KDE\"], training_data::AbstractArray{tp, 2} = [NaN NaN])\n\nreturn an object of type PARAMS, given the algorithms and some training_data as a matrix.\n\nArguments\n\nalgorithms: Subset of [\"REC\", \"KDE\", \"KNN_Gamma\", \"KNN_Delta\", \"SVDD\", \"KNFST\", \"T2\"]\ntraining_data: data for training the algorithms / for getting the Parameters.\ndist::String = \"Euclidean\"\nsigma_quantile::Float64 = 0.5 (median): quantile of the distance matrix, used to compute the weighting parameter for the kernel matrix (algorithms = [\"SVDD\", \"KNFST\", \"KDE\"])\nvarepsilon_quantile = sigma_quantile by default: quantile of the distance matrix to compute the radius of the hyperball in which the number of reccurences is counted (algorihtms = [\"REC\"])\nk_perc::Float64 = 0.05: percentage of the first dimension of training_data to estimmate the number of nearest neighbors (algorithms = [\"KNN-Gamma\", \"KNN_Delta\"])\nnu::Float64 = 0.2: use the maximal percentage of outliers for algorithms = [\"SVDD\"]\ntemp_excl::Int64 = 0. Exclude temporal adjacent points from beeing count as recurrences of k-nearest neighbors algorithms = [\"REC\", \"KNN-Gamma\", \"KNN_Delta\"]\nensemble_method = \"None\": compute an ensemble of the used algorithms. Possible choices (given in compute_ensemble()) are \"mean\", \"median\", \"max\" and \"min\".\nquantiles = false: convert the output scores of the algorithms into quantiles.\n\nExamples\n\njulia> using MultivariateAnomalies\njulia> training_data = randn(100, 2); testing_data = randn(100, 2);\njulia> P = getParameters([\"REC\", \"KDE\", \"SVDD\"], training_data, quantiles = false);\njulia> detectAnomalies(testing_data, P)\n\n\n\n"
},

{
    "location": "man/HighLevelFunctions.html#MultivariateAnomalies.detectAnomalies",
    "page": "High Level Functions",
    "title": "MultivariateAnomalies.detectAnomalies",
    "category": "Function",
    "text": "detectAnomalies{tp, N}(data::AbstractArray{tp, N}, P::PARAMS)\ndetectAnomalies{tp, N}(data::AbstractArray{tp, N}, algorithms::Array{String,1} = [\"REC\", \"KDE\"]; mean = 0)\n\ndetect anomalies, given some Parameter object P of type PARAMS. Train the Parameters P with getParameters() beforehand on some training data. See getParameters(). Without training P beforehand, it is also possible to use detectAnomalies(data, algorithms) given some algorithms (except SVDD, KNFST). Some default parameters are used in this case to initialize P internally.\n\nExamples\n\njulia> training_data = randn(100, 2); testing_data = randn(100, 2);\njulia> # compute the anoamly scores of the algorithms \"REC\", \"KDE\", \"T2\" and \"KNN_Gamma\", their quantiles and return their ensemble scores\njulia> P = getParameters([\"REC\", \"KDE\", \"T2\", \"KNN_Gamma\"], training_data, quantiles = true, ensemble_method = \"mean\");\njulia> detectAnomalies(testing_data, P)\n\n\n\n"
},

{
    "location": "man/HighLevelFunctions.html#MultivariateAnomalies.detectAnomalies!",
    "page": "High Level Functions",
    "title": "MultivariateAnomalies.detectAnomalies!",
    "category": "Function",
    "text": "detectAnomalies!{tp, N}(data::AbstractArray{tp, N}, P::PARAMS)\n\nmutating version of detectAnomalies(). Directly writes the output into P.\n\n\n\n"
},

{
    "location": "man/HighLevelFunctions.html#MultivariateAnomalies.init_detectAnomalies",
    "page": "High Level Functions",
    "title": "MultivariateAnomalies.init_detectAnomalies",
    "category": "Function",
    "text": "init_detectAnomalies{tp, N}(data::AbstractArray{tp, N}, P::PARAMS)\n\ninitialize empty arrays in P for detecting the anomalies.\n\n\n\n"
},

{
    "location": "man/HighLevelFunctions.html#Functions-1",
    "page": "High Level Functions",
    "title": "Functions",
    "category": "section",
    "text": "getParameters\ndetectAnomalies\ndetectAnomalies!\ninit_detectAnomalies"
},

{
    "location": "man/HighLevelFunctions.html#Index-1",
    "page": "High Level Functions",
    "title": "Index",
    "category": "section",
    "text": "Pages = [\"HighLevelFunctions.md\"]"
},

{
    "location": "man/DetectionAlgorithms.html#",
    "page": "Anomaly Detection Algorithms",
    "title": "Anomaly Detection Algorithms",
    "category": "page",
    "text": ""
},

{
    "location": "man/DetectionAlgorithms.html#Anomaly-Detection-Algorithms-1",
    "page": "Anomaly Detection Algorithms",
    "title": "Anomaly Detection Algorithms",
    "category": "section",
    "text": "Most of the anomaly detection algorithms below work on a distance/similarity matrix D or a kernel/dissimilarity matrix K. They can be comuted using the functions provided here.Currently supported algorithms includeRecurrences (REC)\nKernel Density Estimation (KDE)\nHotelling's T^2 (Mahalanobis distance) (T2)\ntwo k-Nearest Neighbor approaches (KNN-Gamma, KNN-Delta)  \nUnivariate Approach (UNIV)\nSupport Vector Data Description (SVDD)\nKernel Null Foley Summon Transform (KNFST)"
},

{
    "location": "man/DetectionAlgorithms.html#Functions-1",
    "page": "Anomaly Detection Algorithms",
    "title": "Functions",
    "category": "section",
    "text": ""
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.REC",
    "page": "Anomaly Detection Algorithms",
    "title": "MultivariateAnomalies.REC",
    "category": "Function",
    "text": "REC(D::AbstractArray, rec_threshold::Float64, temp_excl::Int = 5)\n\nCount the number of observations (recurrences) which fall into a radius rec_threshold of a distance matrix D. Exclude steps which are closer than temp_excl to be count as recurrences (default: temp_excl = 5)\n\nMarwan, N., Carmen Romano, M., Thiel, M., & Kurths, J. (2007). Recurrence plots for the analysis of complex systems. Physics Reports, 438(5-6), 237–329. http://doi.org/10.1016/j.physrep.2006.11.001\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.REC!",
    "page": "Anomaly Detection Algorithms",
    "title": "MultivariateAnomalies.REC!",
    "category": "Function",
    "text": "REC!(rec_out::AbstractArray, D::AbstractArray, rec_threshold::Float64, temp_excl::Int = 5)\n\nMemory efficient version of REC() for use within a loop. rec_out is preallocated output, should be initialised with init_REC().\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.init_REC",
    "page": "Anomaly Detection Algorithms",
    "title": "MultivariateAnomalies.init_REC",
    "category": "Function",
    "text": "init_REC(D::Array{Float64, 2})\ninit_REC(T::Int)\n\nget object for memory efficient REC!() versions. Input can be a distance matrix D or the number of timesteps (observations) T.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#Recurrences-1",
    "page": "Anomaly Detection Algorithms",
    "title": "Recurrences",
    "category": "section",
    "text": "REC\nREC!\ninit_REC"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.KDE",
    "page": "Anomaly Detection Algorithms",
    "title": "MultivariateAnomalies.KDE",
    "category": "Function",
    "text": "KDE(K)\n\nCompute a Kernel Density Estimation (the Parzen sum), given a Kernel matrix K.\n\nParzen, E. (1962). On Estimation of a Probability Density Function and Mode. The Annals of Mathematical Statistics, 33, 1–1065–1076.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.KDE!",
    "page": "Anomaly Detection Algorithms",
    "title": "MultivariateAnomalies.KDE!",
    "category": "Function",
    "text": "KDE!(KDE_out, K)\n\nMemory efficient version of KDE(). Additionally uses preallocated KDE_out object for writing the results. Initialize KDE_out with init_KDE().\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.init_KDE",
    "page": "Anomaly Detection Algorithms",
    "title": "MultivariateAnomalies.init_KDE",
    "category": "Function",
    "text": "init_KDE(K::Array{Float64, 2})\ninit_KDE(T::Int)\n\nReturns KDE_out object for usage in KDE!(). Use either a Kernel matrix K or the number of time steps/observations T as argument.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#Kernel-Density-Estimation-1",
    "page": "Anomaly Detection Algorithms",
    "title": "Kernel Density Estimation",
    "category": "section",
    "text": "KDE\nKDE!\ninit_KDE"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.T2",
    "page": "Anomaly Detection Algorithms",
    "title": "MultivariateAnomalies.T2",
    "category": "Function",
    "text": "T2{tp}(data::AbstractArray{tp,2}, Q::AbstractArray[, mv])\n\nCompute Hotelling's T2 control chart (the squared Mahalanobis distance to the data's mean vector (mv), given the covariance matrix Q). Input data is a two dimensional data matrix (observations * variables).\n\nLowry, C. A., & Woodall, W. H. (1992). A Multivariate Exponentially Weighted Moving Average Control Chart. Technometrics, 34, 46–53.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.T2!",
    "page": "Anomaly Detection Algorithms",
    "title": "MultivariateAnomalies.T2!",
    "category": "Function",
    "text": "T2!(t2_out, data, Q[, mv])\n\nMemory efficient version of T2(), for usage within a loop etc. Initialize the t2_out object with init_T2(). t2_out[1] contains the squred Mahalanobis distance after computation.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.init_T2",
    "page": "Anomaly Detection Algorithms",
    "title": "MultivariateAnomalies.init_T2",
    "category": "Function",
    "text": "init_T2(VAR::Int, T::Int)\ninit_T2{tp}(data::AbstractArray{tp,2})\n\ninitialize t2_out object for T2! either with number of variables VAR and observations/time steps T or with a two dimensional data matrix (time * variables)\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#Hotelling's-Tsup2/sup-1",
    "page": "Anomaly Detection Algorithms",
    "title": "Hotelling's T<sup>2</sup>",
    "category": "section",
    "text": "T2\nT2!\ninit_T2"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.KNN_Gamma",
    "page": "Anomaly Detection Algorithms",
    "title": "MultivariateAnomalies.KNN_Gamma",
    "category": "Function",
    "text": "KNN_Gamma(knn_dists_out)\n\nThis function computes the mean distance of the K nearest neighbors given a knn_dists_out object from knn_dists() as input argument.\n\nHarmeling, S., Dornhege, G., Tax, D., Meinecke, F., & Müller, K.-R. (2006). From outliers to prototypes: Ordering data. Neurocomputing, 69(13-15), 1608–1618. http://doi.org/10.1016/j.neucom.2005.05.015\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.KNN_Gamma!",
    "page": "Anomaly Detection Algorithms",
    "title": "MultivariateAnomalies.KNN_Gamma!",
    "category": "Function",
    "text": "KNN_Gamma!(KNN_Gamma_out, knn_dists_out)\n\nMemory efficient version of KNN_Gamma, to be used in a loop. Initialize KNN_Gamma_out with init_KNN_Gamma().\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.init_KNN_Gamma",
    "page": "Anomaly Detection Algorithms",
    "title": "MultivariateAnomalies.init_KNN_Gamma",
    "category": "Function",
    "text": "init_KNN_Gamma(T::Int)\ninit_KNN_Gamma(knn_dists_out)\n\ninitialize a KNN_Gamma_out object for KNN_Gamma! either with T, the number of observations/time steps or with a knn_dists_out object.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.KNN_Delta",
    "page": "Anomaly Detection Algorithms",
    "title": "MultivariateAnomalies.KNN_Delta",
    "category": "Function",
    "text": "KNN_Delta(knn_dists_out, data)\n\nCompute Delta as vector difference of the k-nearest neighbors. Arguments are a knn_dists() object (knn_dists_out) and a data matrix (observations * variables)\n\nHarmeling, S., Dornhege, G., Tax, D., Meinecke, F., & Müller, K.-R. (2006). From outliers to prototypes: Ordering data. Neurocomputing, 69(13-15), 1608–1618. http://doi.org/10.1016/j.neucom.2005.05.015\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.KNN_Delta!",
    "page": "Anomaly Detection Algorithms",
    "title": "MultivariateAnomalies.KNN_Delta!",
    "category": "Function",
    "text": "KNN_Delta!(KNN_Delta_out, knn_dists_out, data)\n\nMemory Efficient Version of KNN_Delta(). KNN_Delta_out[1] is the vector difference of the k-nearest neighbors.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.init_KNN_Delta",
    "page": "Anomaly Detection Algorithms",
    "title": "MultivariateAnomalies.init_KNN_Delta",
    "category": "Function",
    "text": "init_KNN_Delta(T, VAR, k)\n\nreturn a KNN_Delta_out object to be used for KNN_Delta!. Input: time steps/observations T, variables VAR, number of K nearest neighbors k.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#k-Nearest-Neighbors-1",
    "page": "Anomaly Detection Algorithms",
    "title": "k-Nearest Neighbors",
    "category": "section",
    "text": "KNN_Gamma\nKNN_Gamma!\ninit_KNN_Gamma\nKNN_Delta\nKNN_Delta!\ninit_KNN_Delta"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.UNIV",
    "page": "Anomaly Detection Algorithms",
    "title": "MultivariateAnomalies.UNIV",
    "category": "Function",
    "text": "UNIV(data)\n\norder the values in each varaible and return their maximum, i.e. any of the variables in data (observations * variables) is above a given quantile, the highest quantile will be returned.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.UNIV!",
    "page": "Anomaly Detection Algorithms",
    "title": "MultivariateAnomalies.UNIV!",
    "category": "Function",
    "text": "UNIV!(univ_out, data)\n\nMemory efficient version of UNIV(), input an univ_out object from init_UNIV() and some data matrix observations * variables\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.init_UNIV",
    "page": "Anomaly Detection Algorithms",
    "title": "MultivariateAnomalies.init_UNIV",
    "category": "Function",
    "text": "init_UNIV(T::Int, VAR::Int)\ninit_UNIV{tp}(data::AbstractArray{tp, 2})\n\ninitialize a univ_out object to be used in UNIV!() either with number of time steps/observations T and variables VAR or with a data matrix observations * variables.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#Univariate-Approach-1",
    "page": "Anomaly Detection Algorithms",
    "title": "Univariate Approach",
    "category": "section",
    "text": "UNIV\nUNIV!\ninit_UNIV"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.SVDD_train",
    "page": "Anomaly Detection Algorithms",
    "title": "MultivariateAnomalies.SVDD_train",
    "category": "Function",
    "text": "SVDD_train(K, nu)\n\ntrain a one class support vecort machine model (i.e. support vector data description), given a kernel matrix K and and the highest possible percentage of outliers nu. Returns the model object (svdd_model). Requires LIBSVM.\n\nTax, D. M. J., & Duin, R. P. W. (1999). Support vector domain description. Pattern Recognition Letters, 20, 1191–1199. Schölkopf, B., Williamson, R. C., & Bartlett, P. L. (2000). New Support Vector Algorithms. Neural Computation, 12, 1207–1245.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.SVDD_predict",
    "page": "Anomaly Detection Algorithms",
    "title": "MultivariateAnomalies.SVDD_predict",
    "category": "Function",
    "text": "SVDD_predict(svdd_model, K)\n\npredict the outlierness of an object given the testing Kernel matrix K and the svdd_model from SVDD_train(). Requires LIBSVM.\n\nTax, D. M. J., & Duin, R. P. W. (1999). Support vector domain description. Pattern Recognition Letters, 20, 1191–1199. Schölkopf, B., Williamson, R. C., & Bartlett, P. L. (2000). New Support Vector Algorithms. Neural Computation, 12, 1207–1245.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.SVDD_predict!",
    "page": "Anomaly Detection Algorithms",
    "title": "MultivariateAnomalies.SVDD_predict!",
    "category": "Function",
    "text": "SVDD_predict!(SVDD_out, svdd_model, K)\n\nMemory efficient version of SVDD_predict(). Additional input argument is the SVDD_out object from init_SVDD_predict(). Compute Kwith kernel_matrix(). SVDD_out[1] are predicted labels, SVDD_out[2] decision_values. Requires LIBSVM.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.init_SVDD_predict",
    "page": "Anomaly Detection Algorithms",
    "title": "MultivariateAnomalies.init_SVDD_predict",
    "category": "Function",
    "text": "init_SVDD_predict(T::Int)\ninit_SVDD_predict(T::Int, Ttrain::Int)\n\ninitializes a SVDD_out object to be used in SVDD_predict!(). Input is the number of time steps T (in prediction mode). If T for prediction differs from T of the training data (Ttrain) use Ttrain as additional argument.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#Support-Vector-Data-Description-1",
    "page": "Anomaly Detection Algorithms",
    "title": "Support Vector Data Description",
    "category": "section",
    "text": "SVDD_train\nSVDD_predict\nSVDD_predict!\ninit_SVDD_predict"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.KNFST_train",
    "page": "Anomaly Detection Algorithms",
    "title": "MultivariateAnomalies.KNFST_train",
    "category": "Function",
    "text": "KNFST_train(K)\n\ntrain a one class novelty KNFST model on a Kernel matrix K according to Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler: \"Kernel Null Space Methods for Novelty Detection\". Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013.\n\nOutput\n\n(proj, targetValue) proj 	– projection vector for data points (project x via kx*proj, where kx is row vector containing kernel values of x and training data) targetValue – value of all training samples in the null space\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.KNFST_predict",
    "page": "Anomaly Detection Algorithms",
    "title": "MultivariateAnomalies.KNFST_predict",
    "category": "Function",
    "text": "KNFST_predict(model, K)\n\npredict the outlierness of some data (represented by the kernel matrix K), given some KNFST model from KNFST_train(K). Compute Kwith kernel_matrix().\n\nPaul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler: \"Kernel Null Space Methods for Novelty Detection\". Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.KNFST_predict!",
    "page": "Anomaly Detection Algorithms",
    "title": "MultivariateAnomalies.KNFST_predict!",
    "category": "Function",
    "text": "KNFST_predict!(KNFST_out, KNFST_mod, K)\n\npredict the outlierness of some data (represented by the kernel matrix K), given a KNFST_out object (init_KNFST()), some KNFST model (KNFST_mod = KNFST_train(K)) and the testing kernel matrix K.\n\nPaul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler: \"Kernel Null Space Methods for Novelty Detection\". Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.init_KNFST",
    "page": "Anomaly Detection Algorithms",
    "title": "MultivariateAnomalies.init_KNFST",
    "category": "Function",
    "text": "init_KNFST(T, KNFST_mod)\n\ninitialize a KNFST_outobject for the use with KNFST_predict!, given T, the number of observations and the model output KNFST_train(K).\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#Kernel-Null-Foley-Summon-Transform-1",
    "page": "Anomaly Detection Algorithms",
    "title": "Kernel Null Foley Summon Transform",
    "category": "section",
    "text": "KNFST_train\nKNFST_predict\nKNFST_predict!\ninit_KNFST  "
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.Dist2Centers",
    "page": "Anomaly Detection Algorithms",
    "title": "MultivariateAnomalies.Dist2Centers",
    "category": "Function",
    "text": "Dist2Centers{tp}(centers::AbstractArray{tp, 2})\n\nCompute the distance to the nearest centers of i.e. a K-means clustering output. Large Distances to the nearest center are anomalies. data: Observations * Variables.\n\nExample\n\n(proj, targetValue)\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#Distance-to-some-Centers-1",
    "page": "Anomaly Detection Algorithms",
    "title": "Distance to some Centers",
    "category": "section",
    "text": "Dist2Centers"
},

{
    "location": "man/DetectionAlgorithms.html#Index-1",
    "page": "Anomaly Detection Algorithms",
    "title": "Index",
    "category": "section",
    "text": "Pages = [\"DetectionAlgorithms\"]"
},

{
    "location": "man/DistancesDensity.html#",
    "page": "Distance and Densities",
    "title": "Distance and Densities",
    "category": "page",
    "text": ""
},

{
    "location": "man/DistancesDensity.html#Distance,-Kernel-Matrices-and-k-Nearest-Neighbours-1",
    "page": "Distance and Densities",
    "title": "Distance, Kernel Matrices and k-Nearest Neighbours",
    "category": "section",
    "text": "Compute distance matrices (similarity matrices) and convert them into kernel matrices or k-nearest neighbor objects."
},

{
    "location": "man/DistancesDensity.html#Distance/Similarity-Matrices-1",
    "page": "Distance and Densities",
    "title": "Distance/Similarity Matrices",
    "category": "section",
    "text": "A distance matrix D consists of pairwise distances d()computed with some metrix (e.g. Euclidean):D = d(X_t_i X_t_j)i.e. the distance between vector X of observation t_i and t_j for all observations t_it_j = 1 ldots T."
},

{
    "location": "man/DistancesDensity.html#MultivariateAnomalies.dist_matrix",
    "page": "Distance and Densities",
    "title": "MultivariateAnomalies.dist_matrix",
    "category": "Function",
    "text": "dist_matrix{tp, N}(data::AbstractArray{tp, N}; dist::String = \"Euclidean\", space::Int = 0, lat::Int = 0, lon::Int = 0, Q = 0)\ndist_matrix{tp, N}(data::AbstractArray{tp, N}, training_data; dist::String = \"Euclidean\", space::Int = 0, lat::Int = 0, lon::Int = 0, Q = 0)\n\ncompute the distance matrix of data or the distance matrix between data and training data i.e. the pairwise distances along the first dimension of data, using the last dimension as variables. dist is a distance metric, currently Euclidean(default), SqEuclidean, Chebyshev, Cityblock, JSDivergence, Mahalanobis and SqMahalanobis are supported. The latter two need a covariance matrix Q as input argument.\n\nExamples\n\njulia> dc = randn(10, 4,3)\njulia> D = dist_matrix(dc, space = 2)\n\n\n\n"
},

{
    "location": "man/DistancesDensity.html#MultivariateAnomalies.dist_matrix!",
    "page": "Distance and Densities",
    "title": "MultivariateAnomalies.dist_matrix!",
    "category": "Function",
    "text": "dist_matrix!(D_out, data, ...)\n\ncompute the distance matrix of data, similar to dist_matrix(). D_out object has to be preallocated, i.e. with init_dist_matrix.\n\nExamples\n\njulia> dc = randn(10,4, 4,3)\njulia> D_out = init_dist_matrix(dc)\njulia> dist_matrix!(D_out, dc, lat = 2, lon = 2)\njulia> D_out[1]\n\n\n\n"
},

{
    "location": "man/DistancesDensity.html#MultivariateAnomalies.init_dist_matrix",
    "page": "Distance and Densities",
    "title": "MultivariateAnomalies.init_dist_matrix",
    "category": "Function",
    "text": "init_dist_matrix(data)\ninit_dist_matrix(data, training_data)\n\ninitialize a D_out object for dist_matrix!().\n\n\n\n"
},

{
    "location": "man/DistancesDensity.html#Functions-1",
    "page": "Distance and Densities",
    "title": "Functions",
    "category": "section",
    "text": "dist_matrix\ndist_matrix!\ninit_dist_matrix"
},

{
    "location": "man/DistancesDensity.html#k-Nearest-Neighbor-Objects-1",
    "page": "Distance and Densities",
    "title": "k-Nearest Neighbor Objects",
    "category": "section",
    "text": "k-Nearest Neighbor objects return the k nearest points and their distance out of a distance matrix D."
},

{
    "location": "man/DistancesDensity.html#MultivariateAnomalies.knn_dists",
    "page": "Distance and Densities",
    "title": "MultivariateAnomalies.knn_dists",
    "category": "Function",
    "text": "knn_dists(D, k::Int, temp_excl::Int = 5)\n\nreturns the k-nearest neighbors of a distance matrix D. Excludes temp_excl (default: temp_excl = 5) distances from the main diagonal of D to be also nearest neighbors.\n\nExamples\n\njulia> dc = randn(20, 4,3)\njulia> D = dist_matrix(dc, space = 2)\njulia> knn_dists_out = knn_dists(D, 3, 1)\njulia> knn_dists_out[5] # distances\njulia> knn_dists_out[4] # indices\n\n\n\n"
},

{
    "location": "man/DistancesDensity.html#MultivariateAnomalies.knn_dists!",
    "page": "Distance and Densities",
    "title": "MultivariateAnomalies.knn_dists!",
    "category": "Function",
    "text": "knn_dists!(knn_dists_out, D, temp_excl::Int = 5)\n\nreturns the k-nearest neighbors of a distance matrix D. Similar to knn_dists(), but uses preallocated input object knn_dists_out, initialized with init_knn_dists(). Please note that the number of nearest neighbors k is not necessary, as it is already determined by the knn_dists_out object.\n\nExamples\n\njulia> dc = randn(20, 4,3)\njulia> D = dist_matrix(dc, space = 2)\njulia> knn_dists_out = init_knn_dists(dc, 3)\njulia> knn_dists!(knn_dists_out, D)\njulia> knn_dists_out[5] # distances\njulia> knn_dists_out[4] # indices\n\n\n\n"
},

{
    "location": "man/DistancesDensity.html#MultivariateAnomalies.init_knn_dists",
    "page": "Distance and Densities",
    "title": "MultivariateAnomalies.init_knn_dists",
    "category": "Function",
    "text": "init_knn_dists(T::Int, k::Int)\ninit_knn_dists(datacube::AbstractArray, k::Int)\n\ninitialize a preallocated knn_dists_out object. kis the number of nerarest neighbors, T the number of time steps (i.e. size of the first dimension) or a multidimensional datacube.\n\n\n\n"
},

{
    "location": "man/DistancesDensity.html#Functions-2",
    "page": "Distance and Densities",
    "title": "Functions",
    "category": "section",
    "text": "knn_dists\nknn_dists!\ninit_knn_dists"
},

{
    "location": "man/DistancesDensity.html#Kernel-Matrices-(Dissimilarities)-1",
    "page": "Distance and Densities",
    "title": "Kernel Matrices (Dissimilarities)",
    "category": "section",
    "text": "A distance matrix D can be converted into a kernel matrix K, i.e. by computing pairwise dissimilarities using Gaussian kernels centered on each datapoint. K= exp(-05 cdot D cdot sigma^-2)"
},

{
    "location": "man/DistancesDensity.html#MultivariateAnomalies.kernel_matrix",
    "page": "Distance and Densities",
    "title": "MultivariateAnomalies.kernel_matrix",
    "category": "Function",
    "text": "kernel_matrix(D::AbstractArray, σ::Float64 = 1.0[, kernel::String = \"gauss\", dimension::Int64 = 1])\n\ncompute a kernel matrix out of distance matrix D, given σ. Optionally normalized by the dimension, if kernel = \"normalized_gauss\". compute D with dist_matrix().\n\nExamples\n\njulia> dc = randn(20, 4,3)\njulia> D = dist_matrix(dc, space = 2)\njulia> K = kernel_matrix(D, 2.0)\n\n\n\n"
},

{
    "location": "man/DistancesDensity.html#MultivariateAnomalies.kernel_matrix!",
    "page": "Distance and Densities",
    "title": "MultivariateAnomalies.kernel_matrix!",
    "category": "Function",
    "text": "kernel_matrix!(K, D::AbstractArray, σ::Float64 = 1.0[, kernel::String = \"gauss\", dimension::Int64 = 1])\n\ncompute a kernel matrix out of distance matrix D. Similar to kernel_matrix(), but with preallocated Array K (K = similar(D)) for output.\n\nExamples\n\njulia> dc = randn(20, 4,3)\njulia> D = dist_matrix(dc, space = 2)\njulia> kernel_matrix!(D, D, 2.0) # overwrites distance matrix\n\n\n\n"
},

{
    "location": "man/DistancesDensity.html#Functions-3",
    "page": "Distance and Densities",
    "title": "Functions",
    "category": "section",
    "text": "kernel_matrix\nkernel_matrix!"
},

{
    "location": "man/DistancesDensity.html#Index-1",
    "page": "Distance and Densities",
    "title": "Index",
    "category": "section",
    "text": "Pages = [\"DistancesDensity.md\"]"
},

{
    "location": "man/Postprocessing.html#",
    "page": "Postprocessing",
    "title": "Postprocessing",
    "category": "page",
    "text": ""
},

{
    "location": "man/Postprocessing.html#Scores-1",
    "page": "Postprocessing",
    "title": "Scores",
    "category": "section",
    "text": "Postprocess your anomaly scores by making different algorithms comparable and computing their ensemble."
},

{
    "location": "man/Postprocessing.html#MultivariateAnomalies.get_quantile_scores",
    "page": "Postprocessing",
    "title": "MultivariateAnomalies.get_quantile_scores",
    "category": "Function",
    "text": "get_quantile_scores(scores, quantiles = 0.0:0.01:1.0)\n\nreturn the quantiles of the given N dimensional anomaly scores cube. quantiles (default: quantiles = 0.0:0.01:1.0) is a Float range of quantiles. Any score being greater or equal quantiles[i] and beeing smaller than quantiles[i+1] is assigned to the respective quantile quantiles[i].\n\nExamples\n\njulia> scores1 = rand(10, 2)\njulia> quantile_scores1 = get_quantile_scores(scores1)\n\n\n\n"
},

{
    "location": "man/Postprocessing.html#MultivariateAnomalies.get_quantile_scores!",
    "page": "Postprocessing",
    "title": "MultivariateAnomalies.get_quantile_scores!",
    "category": "Function",
    "text": "get_quantile_scores!{tp,N}(quantile_scores::AbstractArray{Float64, N}, scores::AbstractArray{tp,N}, quantiles::FloatRange{Float64} = 0.0:0.01:1.0)\n\nreturn the quantiles of the given N dimensional scores array into a preallocated quantile_scores array, see get_quantile_scores().\n\n\n\n"
},

{
    "location": "man/Postprocessing.html#MultivariateAnomalies.compute_ensemble",
    "page": "Postprocessing",
    "title": "MultivariateAnomalies.compute_ensemble",
    "category": "Function",
    "text": "compute_ensemble(m1_scores, m2_scores[, m3_scores, m4_scores]; ensemble = \"mean\")\n\ncompute the mean (ensemble = \"mean\"), minimum (ensemble = \"min\"), maximum (ensemble = \"max\") or median (ensemble = \"median\") of the given anomaly scores. Supports between 2 and 4 scores input arrays (m1_scores, ..., m4_scores). The scores of the different anomaly detection algorithms should be somehow comparable, e.g., by using get_quantile_scores() before.\n\nExamples\n\njulia> using MultivariateAnomalies\njulia> scores1 = rand(10, 2)\njulia> scores2 = rand(10, 2)\njulia> quantile_scores1 = get_quantile_scores(scores1)\njulia> quantile_scores2 = get_quantile_scores(scores2)\njulia> compute_ensemble(quantile_scores1, quantile_scores2, ensemble = \"max\")\n\n\n\n"
},

{
    "location": "man/Postprocessing.html#Functions-1",
    "page": "Postprocessing",
    "title": "Functions",
    "category": "section",
    "text": "get_quantile_scores\nget_quantile_scores!\ncompute_ensemble"
},

{
    "location": "man/Postprocessing.html#Index-1",
    "page": "Postprocessing",
    "title": "Index",
    "category": "section",
    "text": "Pages = [\"Postprocessing.md\"]"
},

{
    "location": "man/Preprocessing.html#",
    "page": "Preprocessing",
    "title": "Preprocessing",
    "category": "page",
    "text": ""
},

{
    "location": "man/Preprocessing.html#Feature-Extraction-Techniques-1",
    "page": "Preprocessing",
    "title": "Feature Extraction Techniques",
    "category": "section",
    "text": "Extract the relevant inforamtion out of your data and use them as input feature for the anomaly detection algorithms."
},

{
    "location": "man/Preprocessing.html#Dimensionality-Reduction-1",
    "page": "Preprocessing",
    "title": "Dimensionality Reduction",
    "category": "section",
    "text": "Currently two dimenionality reduction techniques are implemented from MultivariateStats.jl:Principal Component Analysis (PCA)\nIndependent Component Analysis (ICA)"
},

{
    "location": "man/Preprocessing.html#MultivariateAnomalies.globalPCA",
    "page": "Preprocessing",
    "title": "MultivariateAnomalies.globalPCA",
    "category": "Function",
    "text": "globalPCA{tp, N}(datacube::Array{tp, N}, expl_var::Float64 = 0.95)\n\nreturn an orthogonal subset of the variables, i.e. the last dimension of the datacube. A Principal Component Analysis is performed on the entire datacube, explaining at least expl_var of the variance.\n\n\n\n"
},

{
    "location": "man/Preprocessing.html#MultivariateAnomalies.globalICA",
    "page": "Preprocessing",
    "title": "MultivariateAnomalies.globalICA",
    "category": "Function",
    "text": "globalICA(datacube::Array{tp, 4}, mode = \"expl_var\"; expl_var::Float64 = 0.95, num_comp::Int = 3)\n\nperform an Independent Component Analysis on the entire 4-dimensional datacube either by (mode = \"num_comp\") returning num_comp number of independent components or (mode = \"expl_var\") returning the number of components which is necessary to explain expl_var of the variance, when doing a Prinicpal Component Analysis before.\n\n\n\n"
},

{
    "location": "man/Preprocessing.html#Functions-1",
    "page": "Preprocessing",
    "title": "Functions",
    "category": "section",
    "text": "globalPCA\nglobalICA"
},

{
    "location": "man/Preprocessing.html#Seasonality-1",
    "page": "Preprocessing",
    "title": "Seasonality",
    "category": "section",
    "text": "When dealing with time series, i.e. the observations are time steps, it might be important to remove or get robust estimates of the mean seasonal cycles. This is implemended bysubtracting the median seasonal cycle (sMSC) and\ngetting the median seasonal cycle (get_MedianCycles)"
},

{
    "location": "man/Preprocessing.html#MultivariateAnomalies.sMSC",
    "page": "Preprocessing",
    "title": "MultivariateAnomalies.sMSC",
    "category": "Function",
    "text": "sMSC(datacube, cycle_length)\n\nsubtract the median seasonal cycle from the datacube given the length of year cycle_length.\n\nExamples\n\njulia> dc = hcat(rand(193) + 2* sin(0:pi/24:8*pi), rand(193) + 2* sin(0:pi/24:8*pi))\njulia> sMSC_dc = sMSC(dc, 48)\n\n\n\n"
},

{
    "location": "man/Preprocessing.html#MultivariateAnomalies.get_MedianCycles",
    "page": "Preprocessing",
    "title": "MultivariateAnomalies.get_MedianCycles",
    "category": "Function",
    "text": "get_MedianCycles(datacube, cycle_length::Int = 46)\n\nreturns the median annual cycle of a datacube, given the length of the annual cycle (presetting: cycle_length = 46). The datacube can be 2, 3, 4-dimensional, time is stored along the first dimension.\n\nExamples\n\njulia> using MultivariateAnomalies\njulia> dc = hcat(rand(193) + 2* sin(0:pi/24:8*pi), rand(193) + 2* sin(0:pi/24:8*pi))\njulia> cycles = get_MedianCycles(dc, 48)\n\n\n\n"
},

{
    "location": "man/Preprocessing.html#MultivariateAnomalies.get_MedianCycle",
    "page": "Preprocessing",
    "title": "MultivariateAnomalies.get_MedianCycle",
    "category": "Function",
    "text": "get_MedianCycle(dat::Array{tp,1}, cycle_length::Int = 46)\n\nreturns the median annual cycle of a one dimensional data array, given the length of the annual cycle (presetting: cycle_length = 46). Can deal with some NaN values.\n\nExamples\n\njulia> using MultivariateAnomalies\njulia> dat = rand(193) + 2* sin(0:pi/24:8*pi)\njulia> dat[100] = NaN\njulia> cycles = get_MedianCycle(dat, 48)\n\n\n\n"
},

{
    "location": "man/Preprocessing.html#MultivariateAnomalies.get_MedianCycle!",
    "page": "Preprocessing",
    "title": "MultivariateAnomalies.get_MedianCycle!",
    "category": "Function",
    "text": "get_MedianCycle!(init_MC, dat::Array{tp,1})\n\nMemory efficient version of get_MedianCycle(), returning the median cycle in init_MC[3]. The init_MC object should be created with init_MedianCycle. Can deal with some NaN values.\n\nExamples\n\njulia> using MultivariateAnomalies\njulia> dat = rand(193) + 2* sin(0:pi/24:8*pi)\njulia> dat[100] = NaN\njulia> init_MC = init_MedianCycle(dat, 48)\njulia> get_MedianCycle!(init_MC, dat)\njulia> init_MC[3]\n\n\n\n"
},

{
    "location": "man/Preprocessing.html#MultivariateAnomalies.init_MedianCycle",
    "page": "Preprocessing",
    "title": "MultivariateAnomalies.init_MedianCycle",
    "category": "Function",
    "text": "init_MedianCycle(dat::Array{tp}, cycle_length::Int = 46)\ninit_MedianCycle(temporal_length::Int[, cycle_length::Int = 46])\n\ninitialises an init_MC object to be used as input for get_MedianCycle!(). Input is either some sample data or the temporal lenght of the expected input vector and the length of the annual cycle (presetting: cycle_length = 46)\n\n\n\n"
},

{
    "location": "man/Preprocessing.html#Functions-2",
    "page": "Preprocessing",
    "title": "Functions",
    "category": "section",
    "text": "sMSC\nget_MedianCycles\nget_MedianCycle\nget_MedianCycle!\ninit_MedianCycle"
},

{
    "location": "man/Preprocessing.html#Exponential-Weighted-Moving-Average-1",
    "page": "Preprocessing",
    "title": "Exponential Weighted Moving Average",
    "category": "section",
    "text": "One option to reduce the noise level in the data and detect more 'significant' anomalies is computing an exponential weighted moving average (EWMA)"
},

{
    "location": "man/Preprocessing.html#MultivariateAnomalies.EWMA",
    "page": "Preprocessing",
    "title": "MultivariateAnomalies.EWMA",
    "category": "Function",
    "text": "EWMA(dat,  λ)\n\nCompute the exponential weighted moving average (EWMA) with the weighting parameter λ between 0 (full weighting) and 1 (no weighting) along the first dimension of dat. Supports N-dimensional Arrays.\n\nLowry, C. A., & Woodall, W. H. (1992). A Multivariate Exponentially Weighted Moving Average Control Chart. Technometrics, 34, 46–53.\n\nExamples\n\njulia> dc = rand(100,3,2)\njulia> ewma_dc = EWMA(dc, 0.1)\n\n\n\n"
},

{
    "location": "man/Preprocessing.html#MultivariateAnomalies.EWMA!",
    "page": "Preprocessing",
    "title": "MultivariateAnomalies.EWMA!",
    "category": "Function",
    "text": "EWMA!(Z, dat,  λ)\n\nuse a preallocated output Z. Z = similar(dat) or dat = dat for overwriting itself.\n\nExamples\n\njulia> dc = rand(100,3,2)\njulia> EWMA!(dc, dc, 0.1)\n\n\n\n"
},

{
    "location": "man/Preprocessing.html#Function-1",
    "page": "Preprocessing",
    "title": "Function",
    "category": "section",
    "text": "EWMA\nEWMA!"
},

{
    "location": "man/Preprocessing.html#Time-Delay-Embedding-1",
    "page": "Preprocessing",
    "title": "Time Delay Embedding",
    "category": "section",
    "text": "Increase the feature space (Variabales) with lagged observations. "
},

{
    "location": "man/Preprocessing.html#MultivariateAnomalies.TDE",
    "page": "Preprocessing",
    "title": "MultivariateAnomalies.TDE",
    "category": "Function",
    "text": "TDE{tp}(datacube::Array{tp, 4}, ΔT::Integer, DIM::Int = 3)\nTDE{tp}(datacube::Array{tp, 3}, ΔT::Integer, DIM::Int = 3)\n\nreturns an embedded datacube by concatenating lagged versions of the 2-, 3- or 4-dimensional datacube with ΔT time steps in the past up to dimension DIM (presetting: DIM = 3)\n\nExamples\n\njulia> dc = randn(50,3)\njulia> TDE(dc, 3, 2)\n\n\n\n"
},

{
    "location": "man/Preprocessing.html#Function-2",
    "page": "Preprocessing",
    "title": "Function",
    "category": "section",
    "text": "TDE"
},

{
    "location": "man/Preprocessing.html#Moving-Window-Features-1",
    "page": "Preprocessing",
    "title": "Moving Window Features",
    "category": "section",
    "text": "include the variance (mw_VAR) and correlations (mw_COR) in a moving window along the first dimension of the data."
},

{
    "location": "man/Preprocessing.html#MultivariateAnomalies.mw_VAR",
    "page": "Preprocessing",
    "title": "MultivariateAnomalies.mw_VAR",
    "category": "Function",
    "text": "mw_VAR{tp,N}(datacube::Array{tp,N}, windowsize::Int = 10)\n\ncompute the variance in a moving window along the first dimension of the datacube (presetting: windowsize = 10). Accepts N dimensional datacubes.\n\nExamples\n\njulia> dc = randn(50,3,3,3)\njulia> mw_VAR(dc, 15)\n\n\n\n"
},

{
    "location": "man/Preprocessing.html#MultivariateAnomalies.mw_VAR!",
    "page": "Preprocessing",
    "title": "MultivariateAnomalies.mw_VAR!",
    "category": "Function",
    "text": "mw_VAR!{tp,N}(out::Array{tp, N}, datacube0mean::Array{tp,N}, windowsize::Int = 10)\n\nmutating version for mw_VAR(). The mean of the input data datacube0mean has to be 0. Initialize out properly: out = datacube0mean leads to wrong results.\n\n\n\n"
},

{
    "location": "man/Preprocessing.html#MultivariateAnomalies.mw_COR",
    "page": "Preprocessing",
    "title": "MultivariateAnomalies.mw_COR",
    "category": "Function",
    "text": "mw_COR{tp}(datacube::Array{tp, 4}, windowsize::Int = 10)\n\ncompute the correlation in a moving window along the first dimension of the datacube (presetting: windowsize = 10). Accepts 4-dimensional datacubes.\n\n\n\n"
},

{
    "location": "man/Preprocessing.html#MultivariateAnomalies.mw_AVG",
    "page": "Preprocessing",
    "title": "MultivariateAnomalies.mw_AVG",
    "category": "Function",
    "text": "mw_AVG{tp,N}(datacube::AbstractArray{tp,N}, windowsize::Int = 10)\n\ncompute the average in a moving window along the first dimension of the datacube (presetting: windowsize = 10). Accepts N dimensional datacubes.\n\nExamples\n\njulia> dc = randn(50,3,3,3)\njulia> mw_AVG(dc, 15)\n\n\n\n"
},

{
    "location": "man/Preprocessing.html#MultivariateAnomalies.mw_AVG!",
    "page": "Preprocessing",
    "title": "MultivariateAnomalies.mw_AVG!",
    "category": "Function",
    "text": "mw_AVG!{tp,N}(out::Array{tp, N}, datacube::Array{tp,N}, windowsize::Int = 10)\n\ninternal and mutating version for mw_AVG().\n\n\n\n"
},

{
    "location": "man/Preprocessing.html#Functions-3",
    "page": "Preprocessing",
    "title": "Functions",
    "category": "section",
    "text": "mw_VAR\nmw_VAR!\nmw_COR\nmw_AVG\nmw_AVG!"
},

{
    "location": "man/Preprocessing.html#Index-1",
    "page": "Preprocessing",
    "title": "Index",
    "category": "section",
    "text": "Pages = [\"Preprocessing.md\"]"
},

{
    "location": "man/AUC.html#",
    "page": "AUC",
    "title": "AUC",
    "category": "page",
    "text": ""
},

{
    "location": "man/AUC.html#Area-Under-the-Curve-1",
    "page": "AUC",
    "title": "Area Under the Curve",
    "category": "section",
    "text": "Compute true positive rates, false positive rates and the area under the curve to evaulate the algorihtms performance. Efficient implementation according toFawcett, T. (2006). An introduction to ROC analysis. Pattern Recognition Letters, 27(8), 861–874. Link"
},

{
    "location": "man/AUC.html#MultivariateAnomalies.auc",
    "page": "AUC",
    "title": "MultivariateAnomalies.auc",
    "category": "Function",
    "text": "auc(scores, events, increasing = true)\n\ncompute the Area Under the receiver operator Curve (AUC), given some output scores array and some ground truth (events). By default, it is assumed, that the scores are ordered increasingly (increasing = true), i.e. high scores represent events.\n\nExamples\n\njulia> scores = rand(10, 2)\njulia> events = rand(0:1, 10, 2)\njulia> auc(scores, events)\njulia> auc(scores, boolevents(events))\n\n\n\n"
},

{
    "location": "man/AUC.html#MultivariateAnomalies.auc_fpr_tpr",
    "page": "AUC",
    "title": "MultivariateAnomalies.auc_fpr_tpr",
    "category": "Function",
    "text": "auc_fpr_tpr(scores, events, quant = 0.9, increasing = true)\n\nSimilar like auc(), but return additionally the true positive and false positive rate at a given quantile (default: quant = 0.9).\n\nExamples\n\njulia> scores = rand(10, 2)\njulia> events = rand(0:1, 10, 2)\njulia> auc_fpr_tpr(scores, events, 0.8)\n\n\n\n"
},

{
    "location": "man/AUC.html#MultivariateAnomalies.boolevents",
    "page": "AUC",
    "title": "MultivariateAnomalies.boolevents",
    "category": "Function",
    "text": "boolevents(events)\n\nconvert an events array into a boolean array.\n\n\n\n"
},

{
    "location": "man/AUC.html#Functions-1",
    "page": "AUC",
    "title": "Functions",
    "category": "section",
    "text": "auc\nauc_fpr_tpr\nbooleventsIndexPages = [\"AUC.md\"]"
},

{
    "location": "man/OnlineAlgorithms.html#",
    "page": "OnlineAlgorithms",
    "title": "OnlineAlgorithms",
    "category": "page",
    "text": ""
},

{
    "location": "man/OnlineAlgorithms.html#Online-Algorithms-1",
    "page": "OnlineAlgorithms",
    "title": "Online Algorithms",
    "category": "section",
    "text": "We provide online some functions, which are tuned to allocate minimal amounts of memory. Implemented so far: Euclidean distance\nSigma estimation for KDE\nKDE\nREC (in progress)\nKNN-Gamma (in progress)"
},

{
    "location": "man/OnlineAlgorithms.html#MultivariateAnomalies.Euclidean_distance!",
    "page": "OnlineAlgorithms",
    "title": "MultivariateAnomalies.Euclidean_distance!",
    "category": "Function",
    "text": "Euclidean_distance!{tp}(d::Array{tp, 1}, x::AbstractArray{tp, 2}, i::Int, j::Int, dim::Int = 1)\n\ncompute the Euclidean distance between x[i,:] and x[j,:] and write the result to d. Memory efficient. dim is the dimension of i and j.\n\n\n\n"
},

{
    "location": "man/OnlineAlgorithms.html#MultivariateAnomalies.SigmaOnline!",
    "page": "OnlineAlgorithms",
    "title": "MultivariateAnomalies.SigmaOnline!",
    "category": "Function",
    "text": "SigmaOnline!{tp}(sigma::Array{tp, 1}, x::AbstractArray{tp, 2}, samplesize::Int = 250, dim::Int = 1)\n\ncompute sigma parameter as mean of the distances of samplesize randomly sampled points along dim.\n\n\n\n"
},

{
    "location": "man/OnlineAlgorithms.html#MultivariateAnomalies.KDEonline!",
    "page": "OnlineAlgorithms",
    "title": "MultivariateAnomalies.KDEonline!",
    "category": "Function",
    "text": "KDEonline!{tp}(kdescores::AbstractArray{tp, 1}, x::AbstractArray{tp, 2}, σ::tp, dim::Int = 1)\n\ncompute (1.0 - Kernel Density Estimates) from x and write it to kdescores with dim being the dimension of the observations.\n\n\n\n"
},

{
    "location": "man/OnlineAlgorithms.html#Functions-1",
    "page": "OnlineAlgorithms",
    "title": "Functions",
    "category": "section",
    "text": "Euclidean_distance!\nSigmaOnline!\nKDEonline!"
},

{
    "location": "man/OnlineAlgorithms.html#Index-1",
    "page": "OnlineAlgorithms",
    "title": "Index",
    "category": "section",
    "text": "Pages = [\"OnlineAlgorithms.md\"]"
},

]}
