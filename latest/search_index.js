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
    "text": "A julia package for detecting multivariate anomalies.Keywords: Novelty detection, Anomaly Detection, Outlier Detection, Statistical Process ControlPlease cite this package as ..."
},

{
    "location": "index.html#Requirements-1",
    "page": "Home",
    "title": "Requirements",
    "category": "section",
    "text": "Julia \n0.4\nDistances\n, \nMultivariateStats\n latest \nLIBSVM\n branch via: \nPkg.clone(\"https://github.com/milanflach/LIBSVM.jl.git\");\n \nPkg.checkout(\"LIBSVM\", \"mutating_versions\");\n \nPkg.build(\"LIBSVM\")"
},

{
    "location": "index.html#Package-Features-1",
    "page": "Home",
    "title": "Package Features",
    "category": "section",
    "text": "Extract the relevant features from the data  ```@contents Pages = [\"man/FeatureExtraction.md\"] ```\nCompute Distance, Kernel matrices and k-nearest neighbors objects  ```@contents Pages = [\"man/DistDensity.md\"] ```\nDetect the anomalies ```@contents Pages = [\"man/DetectionAlgorithms.md\"] ```\nPostprocess your anomaly scores, by computing their quantiles or ensembles ```@contents Pages = [\"man/Scores.md\"] ```\nCompute the area under the curve as external evaluation metric ```@contents Pages = [\"man/AUC.md\"] ```"
},

{
    "location": "index.html#Using-the-Package-1",
    "page": "Home",
    "title": "Using the Package",
    "category": "section",
    "text": "We provide high-level convenience functions for detecting the anomalies. Namely the pair of P = getParameters(algorithms, training_data) and detectAnomalies(testing_data, P)sets standard choices of the Parameters P and hands the parameters as well as the algorithms choice over to detect the anomalies. Currently supported algorithms include Kernel Density Estimation (algorithms = [\"KDE\"]), Recurrences (\"REC\"), k-Nearest Neighbors algorithms (\"KNN-Gamma\", \"KNN-Delta\"), Hotelling's T^2 (\"T2\"), Support Vector Data Description (\"SVDD\") and Kernel Null Foley Summon Transform (\"KNFST\"). With getParameters() it is also possible to compute output scores of multiple algorithms at once (algorihtms = [\"KDE\", \"T2\"]), quantiles of the output anomaly scores (quantiles = true) and ensembles of the selected algorithms (e.g. ensemble_method = \"mean\"). For more details about the detection algorithms and their usage please consider @contents Pages = [\"man/DetectionAlgorithms.md\"]"
},

{
    "location": "index.html#Input-Data-1",
    "page": "Home",
    "title": "Input Data",
    "category": "section",
    "text": "Within MultivariateAnomalies we assume that observations/samples/time steps are stored along the first dimension of the data array (rows of a matrix) with the number of observations T = size(data, 1). Variables/attributes are stored along the last dimension N of the data array (along the columns of a matrix) with the number of variables VAR = size(data, N). We are interested in the question which observation(s) of the data are anomalous."
},

{
    "location": "index.html#Index-1",
    "page": "Home",
    "title": "Index",
    "category": "section",
    "text": "Pages = [\"man/FeatureExtraction.md\", \"man/DetectionAlgorithms.md\", \"man/Scores.md\", \"man/AUC.md\", \"man/DistDensity.md\"]"
},

{
    "location": "man/FeatureExtraction.html#",
    "page": "-",
    "title": "-",
    "category": "page",
    "text": ""
},

{
    "location": "man/FeatureExtraction.html#Feature-Extraction-Techniques-1",
    "page": "-",
    "title": "Feature Extraction Techniques",
    "category": "section",
    "text": "Extract the relevant inforamtion out of your data and use them as input feature for the anomaly detection algorithms."
},

{
    "location": "man/FeatureExtraction.html#MultivariateAnomalies.sMSC",
    "page": "-",
    "title": "MultivariateAnomalies.sMSC",
    "category": "Function",
    "text": "sMSC(datacube, cycle_length)\n\nsubtract the median seasonal cycle from the datacube given the length of year cycle_length.\n\nExamples\n\njulia> dc = hcat(rand(193) + 2* sin(0:pi/24:8*pi), rand(193) + 2* sin(0:pi/24:8*pi))\njulia> sMSC_dc = sMSC(dc, 48)\n\n\n\n"
},

{
    "location": "man/FeatureExtraction.html#MultivariateAnomalies.globalPCA",
    "page": "-",
    "title": "MultivariateAnomalies.globalPCA",
    "category": "Function",
    "text": "globalPCA{tp, N}(datacube::Array{tp, N}, expl_var::Float64 = 0.95)\n\nreturn an orthogonal subset of the variables, i.e. the last dimension of the datacube. A Principal Component Analysis is performed on the entire datacube, explaining at least expl_var of the variance.\n\n\n\n"
},

{
    "location": "man/FeatureExtraction.html#MultivariateAnomalies.globalICA",
    "page": "-",
    "title": "MultivariateAnomalies.globalICA",
    "category": "Function",
    "text": "globalICA(datacube::Array{tp, 4}, mode = \"expl_var\"; expl_var::Float64 = 0.95, num_comp::Int = 3)\n\nperform an Independent Component Analysis on the entire 4-dimensional datacube either by (mode = \"num_comp\") returning num_comp number of independent components or (mode = \"expl_var\") returning the number of components which is necessary to explain expl_var of the variance, when doing a Prinicpal Component Analysis before.\n\n\n\n"
},

{
    "location": "man/FeatureExtraction.html#MultivariateAnomalies.TDE",
    "page": "-",
    "title": "MultivariateAnomalies.TDE",
    "category": "Function",
    "text": "TDE{tp}(datacube::Array{tp, 4}, ΔT::Integer, DIM::Int = 3)\nTDE{tp}(datacube::Array{tp, 3}, ΔT::Integer, DIM::Int = 3)\n\nreturns an embedded datacube by concatenating lagged versions of the 2-, 3- or 4-dimensional datacube with ΔT time steps in the past up to dimension DIM (presetting: DIM = 3)\n\njulia> dc = randn(50,3)\njulia> TDE(dc, 3, 2)\n\n\n\n"
},

{
    "location": "man/FeatureExtraction.html#MultivariateAnomalies.mw_VAR",
    "page": "-",
    "title": "MultivariateAnomalies.mw_VAR",
    "category": "Function",
    "text": "mw_VAR{tp,N}(datacube::Array{tp,N}, windowsize::Int = 10)\n\ncompute the variance in a moving window along the first dimension of the datacube (presetting: windowsize = 10). Accepts N dimensional datacubes.\n\njulia> dc = randn(50,3,3,3)\njulia> mw_VAR(dc, 15)\n\n\n\n"
},

{
    "location": "man/FeatureExtraction.html#MultivariateAnomalies.mw_COR",
    "page": "-",
    "title": "MultivariateAnomalies.mw_COR",
    "category": "Function",
    "text": "mw_COR{tp}(datacube::Array{tp, 4}, windowsize::Int = 10)\n\ncompute the correlation in a moving window along the first dimension of the datacube (presetting: windowsize = 10). Accepts 4-dimensional datacubes.\n\n\n\n"
},

{
    "location": "man/FeatureExtraction.html#MultivariateAnomalies.EWMA",
    "page": "-",
    "title": "MultivariateAnomalies.EWMA",
    "category": "Function",
    "text": "EWMA(dat,  λ)\n\nCompute the exponential weighted moving average (EWMA) with the weighting parameter λ between 0 (full weighting) and 1 (no weighting) in the first dimension of dat. Supports N-dimensional Arrays.\n\nLowry, C. A., & Woodall, W. H. (1992). A Multivariate Exponentially Weighted Moving Average Control Chart. Technometrics, 34, 46–53.\n\njulia> dc = rand(100,3,2)\njulia> ewma_dc = EWMA(dc, 0.1)\n\n\n\n"
},

{
    "location": "man/FeatureExtraction.html#MultivariateAnomalies.EWMA!",
    "page": "-",
    "title": "MultivariateAnomalies.EWMA!",
    "category": "Function",
    "text": "EWMA!(Z, dat,  λ)\n\nuse a preallocated output Z. Z = similar(dat) or dat = dat for overwriting itself.\n\nExamples\n\njulia> dc = rand(100,3,2)\njulia> EWMA!(dc, dc, 0.1)\n\n\n\n"
},

{
    "location": "man/FeatureExtraction.html#MultivariateAnomalies.get_MedianCycles",
    "page": "-",
    "title": "MultivariateAnomalies.get_MedianCycles",
    "category": "Function",
    "text": "get_MedianCycles(datacube, cycle_length::Int = 46)\n\nreturns the median annual cycle of a datacube, given the length of the annual cycle (presetting: cycle_length = 46). The datacube can be 2, 3, 4-dimensional, time is stored along the first dimension.\n\nExamples\n\njulia> dc = hcat(rand(193) + 2* sin(0:pi/24:8*pi), rand(193) + 2* sin(0:pi/24:8*pi))\njulia> cycles = get_MedianCycles(dc, 48)\n\n\n\n"
},

{
    "location": "man/FeatureExtraction.html#MultivariateAnomalies.get_MedianCycle",
    "page": "-",
    "title": "MultivariateAnomalies.get_MedianCycle",
    "category": "Function",
    "text": "get_MedianCycle(dat::Array{tp,1}, cycle_length::Int = 46)\n\nreturns the median annual cycle of a one dimensional data array, given the length of the annual cycle (presetting: cycle_length = 46). Can deal with some NaN values.\n\nExamples\n\njulia> dat = rand(193) + 2* sin(0:pi/24:8*pi)\njulia> dat[100] = NaN\njulia> cycles = get_MedianCycle(dat, 48)\n\n\n\n"
},

{
    "location": "man/FeatureExtraction.html#MultivariateAnomalies.get_MedianCycle!",
    "page": "-",
    "title": "MultivariateAnomalies.get_MedianCycle!",
    "category": "Function",
    "text": "get_MedianCycle!(init_MC, dat::Array{tp,1})\n\nMemory efficient version of get_MedianCycle(), returning the median cycle in init_MC[3]. The init_MC object should be created with init_MedianCycle. Can deal with some NaN values.\n\nExamples\n\njulia> dat = rand(193) + 2* sin(0:pi/24:8*pi)\njulia> dat[100] = NaN\njulia> init_MC = init_MedianCycle(dat, 48)\njulia> get_MedianCycle!(init_MC, dat)\njulia> init_MC[3]\n\n\n\n"
},

{
    "location": "man/FeatureExtraction.html#MultivariateAnomalies.init_MedianCycle",
    "page": "-",
    "title": "MultivariateAnomalies.init_MedianCycle",
    "category": "Function",
    "text": "init_MedianCycle(dat::Array{tp}, cycle_length::Int = 46)\ninit_MedianCycle(temporal_length::Int[, cycle_length::Int = 46])\n\ninitialises an init_MC object to be used as input for get_MedianCycle!(). Input is either some sample data or the temporal lenght of the expected input vector and the length of the annual cycle (presetting: cycle_length = 46)\n\n\n\n"
},

{
    "location": "man/FeatureExtraction.html#Functions-1",
    "page": "-",
    "title": "Functions",
    "category": "section",
    "text": "sMSC\nglobalPCA\nglobalICA\nTDE\nmw_VAR\nmw_COR\nEWMA\nEWMA!\nget_MedianCycles\nget_MedianCycle\nget_MedianCycle!\ninit_MedianCycle"
},

{
    "location": "man/FeatureExtraction.html#Index-1",
    "page": "-",
    "title": "Index",
    "category": "section",
    "text": ""
},

{
    "location": "man/DetectionAlgorithms.html#",
    "page": "-",
    "title": "-",
    "category": "page",
    "text": ""
},

{
    "location": "man/DetectionAlgorithms.html#Detection-Algorithms-1",
    "page": "-",
    "title": "Detection Algorithms",
    "category": "section",
    "text": "detect anomalies out of multivariate correlated data. "
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.getParameters",
    "page": "-",
    "title": "MultivariateAnomalies.getParameters",
    "category": "Function",
    "text": "getParameters(algorithms::Array{ASCIIString,1} = [\"REC\", \"KDE\"], training_data::AbstractArray{tp, 2} = [NaN NaN])\n\nreturn an object of type PARAMS, given the algorithms and some training_data as a matrix.\n\nArguments\n\nalgorithms\n: Subset of \n[\"REC\", \"KDE\", \"KNN_Gamma\", \"KNN_Delta\", \"SVDD\", \"KNFST\", \"T2\"]\ntraining_data\n: data for training the algorithms / for getting the Parameters.\ndist::ASCIIString = \"Euclidean\"\nsigma_quantile::Float64 = 0.5\n (median): quantile of the distance matrix, used to compute the weighting parameter for the kernel matrix (\nalgorithms = [\"SVDD\", \"KNFST\", \"KDE\"]\n)\nvarepsilon_quantile\n = \nsigma_quantile\n by default: quantile of the distance matrix to compute the radius of the hyperball in which the number of reccurences is counted (\nalgorihtms = [\"REC\"]\n)\nk_perc::Float64 = 0.05\n: percentage of the first dimension of \ntraining_data\n to estimmate the number of nearest neighbors (\nalgorithms = [\"KNN-Gamma\", \"KNN_Delta\"]\n)\nnu::Float64 = 0.2\n: use the maximal percentage of outliers for \nalgorithms = [\"SVDD\"]\ntemp_excl::Int64 = 0\n. Exclude temporal adjacent points from beeing count as recurrences of k-nearest neighbors \nalgorithms = [\"REC\", \"KNN-Gamma\", \"KNN_Delta\"]\nensemble_method = \"None\"\n: compute an ensemble of the used algorithms. Possible choices (given in \ncompute_ensemble()\n) are \"mean\", \"median\", \"max\" and \"min\".\nquantiles = false\n: convert the output scores of the algorithms into quantiles.\n\nExamples\n\njulia> training_data = randn(100, 2); testing_data = randn(100, 2);\njulia> P = getParameters([\"REC\", \"KDE\", \"SVDD\"], training_data, quantiles = false);\njulia> detectAnomalies(testing_data, P)\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.detectAnomalies",
    "page": "-",
    "title": "MultivariateAnomalies.detectAnomalies",
    "category": "Function",
    "text": "detectAnomalies{tp, N}(data::AbstractArray{tp, N}, P::PARAMS)\ndetectAnomalies{tp, N}(data::AbstractArray{tp, N}, algorithms::Array{ASCIIString,1} = [\"REC\", \"KDE\"]; mean = 0)\n\ndetect anomalies, given some Parameter object P of type PARAMS. Train the Parameters P with getParameters() beforehand on some training data. See getParameters(). Without training P beforehand, it is also possible to use detectAnomalies(data, algorithms) given some algorithms (except SVDD, KNFST). Some default parameters are used in this case to initialize P internally.\n\nExamples\n\njulia> training_data = randn(100, 2); testing_data = randn(100, 2);\njulia> # compute the anoamly scores of the algorithms \"REC\", \"KDE\", \"T2\" and \"KNN_Gamma\", their quantiles and return their ensemble scores\njulia> P = getParameters([\"REC\", \"KDE\", \"T2\", \"KNN_Gamma\"], training_data, quantiles = true, ensemble_method = \"mean\");\njulia> detectAnomalies(testing_data, P)\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.detectAnomalies!",
    "page": "-",
    "title": "MultivariateAnomalies.detectAnomalies!",
    "category": "Function",
    "text": "detectAnomalies!{tp, N}(data::AbstractArray{tp, N}, P::PARAMS)\n\nmutating version of detectAnomalies(). Directly writes the output into P.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.init_detectAnomalies",
    "page": "-",
    "title": "MultivariateAnomalies.init_detectAnomalies",
    "category": "Function",
    "text": "init_detectAnomalies{tp, N}(data::AbstractArray{tp, N}, P::PARAMS)\n\ninitialize empty arrays in P for detecting the anomalies.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#High-Level-Functions-1",
    "page": "-",
    "title": "High Level Functions",
    "category": "section",
    "text": "getParameters\ndetectAnomalies\ndetectAnomalies!\ninit_detectAnomalies"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.REC",
    "page": "-",
    "title": "MultivariateAnomalies.REC",
    "category": "Function",
    "text": "REC(D::AbstractArray, rec_threshold::Float64, temp_excl::Int = 5)\n\nCount the number of observations (recurrences) which fall into a radius rec_threshold of a distance matrix D. Exclude steps which are closer than temp_excl to be count as recurrences (default: temp_excl = 5)\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.REC!",
    "page": "-",
    "title": "MultivariateAnomalies.REC!",
    "category": "Function",
    "text": "REC!(rec_out::AbstractArray, D::AbstractArray, rec_threshold::Float64, temp_excl::Int = 5)\n\nMemory efficient version of REC() for use within a loop. rec_out is preallocated output, should be initialised with init_REC().\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.init_REC",
    "page": "-",
    "title": "MultivariateAnomalies.init_REC",
    "category": "Function",
    "text": "init_REC(D::Array{Float64, 2})\ninit_REC(T::Int)\n\nget object for memory efficient REC!() versions. Input can be a distance matrix D or the number of timesteps (observations) T.\n\nMarwan, N., Carmen Romano, M., Thiel, M., & Kurths, J. (2007). Recurrence plots for the analysis of complex systems. Physics Reports, 438(5-6), 237–329. http://doi.org/10.1016/j.physrep.2006.11.001\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.KDE",
    "page": "-",
    "title": "MultivariateAnomalies.KDE",
    "category": "Function",
    "text": "KDE(K)\n\nCompute a Kernel Density Estimation (the Parzen sum), given a Kernel matrix K.\n\nParzen, E. (1962). On Estimation of a Probability Density Function and Mode. The Annals of Mathematical Statistics, 33, 1–1065–1076.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.KDE!",
    "page": "-",
    "title": "MultivariateAnomalies.KDE!",
    "category": "Function",
    "text": "KDE!(KDE_out, K)\n\nMemory efficient version of KDE(). Additionally uses preallocated KDE_out object for writing the results. Initialize KDE_out with init_KDE().\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.init_KDE",
    "page": "-",
    "title": "MultivariateAnomalies.init_KDE",
    "category": "Function",
    "text": "init_KDE(K::Array{Float64, 2})\ninit_KDE(T::Int)\n\nReturns KDE_out object for usage in KDE!(). Use either a Kernel matrix K or the number of time steps/observations T as argument.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.T2",
    "page": "-",
    "title": "MultivariateAnomalies.T2",
    "category": "Function",
    "text": "T2{tp}(data::AbstractArray{tp,2}, Q::AbstractArray[, mv])\n\nCompute Hotelling's T^2 control chart (the squared Mahalanobis distance to the data's mean vector (mv), given the covariance matrix Q). Input data is a two dimensional data matrix (observations * variables).\n\nLowry, C. A., & Woodall, W. H. (1992). A Multivariate Exponentially Weighted Moving Average Control Chart. Technometrics, 34, 46–53.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.T2!",
    "page": "-",
    "title": "MultivariateAnomalies.T2!",
    "category": "Function",
    "text": "T2!(t2_out, data, Q[, mv])\n\nMemory efficient version of T2(), for usage within a loop etc. Initialize the t2_out object with init_T2(). t2_out[1] contains the squred Mahalanobis distance after computation.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.init_T2",
    "page": "-",
    "title": "MultivariateAnomalies.init_T2",
    "category": "Function",
    "text": "init_T2(VAR::Int, T::Int)\ninit_T2{tp}(data::AbstractArray{tp,2})\n\ninitialize t2_out object for T2! either with number of variables VAR and observations/time steps T or with a two dimensional data matrix (time * variables)\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.KNN_Gamma",
    "page": "-",
    "title": "MultivariateAnomalies.KNN_Gamma",
    "category": "Function",
    "text": "KNN_Gamma(knn_dists_out)\n\nThis function computes the mean distance of the K nearest neighbors given a knn_dists_out object from knn_dists() as input argument.\n\nHarmeling, S., Dornhege, G., Tax, D., Meinecke, F., & Müller, K.-R. (2006). From outliers to prototypes: Ordering data. Neurocomputing, 69(13-15), 1608–1618. http://doi.org/10.1016/j.neucom.2005.05.015\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.KNN_Gamma!",
    "page": "-",
    "title": "MultivariateAnomalies.KNN_Gamma!",
    "category": "Function",
    "text": "KNN_Gamma!(KNN_Gamma_out, knn_dists_out)\n\nMemory efficient version of KNN_Gamma, to be used in a loop. Initialize KNN_Gamma_out with init_KNN_Gamma().\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.init_KNN_Gamma",
    "page": "-",
    "title": "MultivariateAnomalies.init_KNN_Gamma",
    "category": "Function",
    "text": "init_KNN_Gamma(T::Int)\ninit_KNN_Gamma(knn_dists_out)\n\ninitialize a KNN_Gamma_out object for KNN_Gamma! either with T, the number of observations/time steps or with a knn_dists_out object.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.KNN_Delta",
    "page": "-",
    "title": "MultivariateAnomalies.KNN_Delta",
    "category": "Function",
    "text": "KNN_Delta(knn_dists_out, data)\n\nCompute Delta as vector difference of the k-nearest neighbors. Arguments are a knn_dists() object (knn_dists_out) and a data matrix (observations * variables)\n\nHarmeling, S., Dornhege, G., Tax, D., Meinecke, F., & Müller, K.-R. (2006). From outliers to prototypes: Ordering data. Neurocomputing, 69(13-15), 1608–1618. http://doi.org/10.1016/j.neucom.2005.05.015\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.KNN_Delta!",
    "page": "-",
    "title": "MultivariateAnomalies.KNN_Delta!",
    "category": "Function",
    "text": "KNN_Delta!(KNN_Delta_out, knn_dists_out, data)\n\nMemory Efficient Version of KNN_Delta(). KNN_Delta_out[1] is the vector difference of the k-nearest neighbors.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.init_KNN_Delta",
    "page": "-",
    "title": "MultivariateAnomalies.init_KNN_Delta",
    "category": "Function",
    "text": "init_KNN_Delta(T, VAR, k)\n\nreturn a KNN_Delta_out object to be used for KNN_Delta!. Input: time steps/observations T, variables VAR, number of K nearest neighbors k.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.UNIV",
    "page": "-",
    "title": "MultivariateAnomalies.UNIV",
    "category": "Function",
    "text": "UNIV(data)\n\norder the values in each varaible and return their maximum, i.e. any of the variables in data (observations * variables) is above a given quantile, the highest quantile will be returned.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.UNIV!",
    "page": "-",
    "title": "MultivariateAnomalies.UNIV!",
    "category": "Function",
    "text": "UNIV!(univ_out, data)\n\nMemory efficient version of UNIV(), input an univ_out object from init_UNIV() and some data matrix observations * variables\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.init_UNIV",
    "page": "-",
    "title": "MultivariateAnomalies.init_UNIV",
    "category": "Function",
    "text": "init_UNIV(T::Int, VAR::Int)\ninit_UNIV{tp}(data::AbstractArray{tp, 2})\n\ninitialize a univ_out object to be used in UNIV!() either with number of time steps/observations T and variables VAR or with a data matrix observations * variables.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.SVDD_train",
    "page": "-",
    "title": "MultivariateAnomalies.SVDD_train",
    "category": "Function",
    "text": "SVDD_train(K, nu)\n\ntrain a one class support vecort machine model (i.e. support vector data description), given a kernel matrix K and and the highest possible percentage of outliers nu. Returns the model object (svdd_model). Requires LIBSVM.\n\nTax, D. M. J., & Duin, R. P. W. (1999). Support vector domain description. Pattern Recognition Letters, 20, 1191–1199. Schölkopf, B., Williamson, R. C., & Bartlett, P. L. (2000). New Support Vector Algorithms. Neural Computation, 12, 1207–1245.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.SVDD_predict",
    "page": "-",
    "title": "MultivariateAnomalies.SVDD_predict",
    "category": "Function",
    "text": "SVDD_predict(svdd_model, K)\n\npredict the outlierness of an object given the testing Kernel matrix K and the svdd_model from SVDD_train(). Requires LIBSVM.\n\nTax, D. M. J., & Duin, R. P. W. (1999). Support vector domain description. Pattern Recognition Letters, 20, 1191–1199. Schölkopf, B., Williamson, R. C., & Bartlett, P. L. (2000). New Support Vector Algorithms. Neural Computation, 12, 1207–1245.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.SVDD_predict!",
    "page": "-",
    "title": "MultivariateAnomalies.SVDD_predict!",
    "category": "Function",
    "text": "SVDD_predict!(SVDD_out, svdd_model, K)\n\nMemory efficient version of SVDD_predict(). Additional input argument is the SVDD_out object from init_SVDD_predict(). Compute Kwith kernel_matrix(). SVDD_out[1] are predicted labels, SVDD_out[2] decision_values. Requires LIBSVM.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.init_SVDD_predict",
    "page": "-",
    "title": "MultivariateAnomalies.init_SVDD_predict",
    "category": "Function",
    "text": "init_SVDD_predict(T::Int)\ninit_SVDD_predict(T::Int, Ttrain::Int)\n\ninitializes a SVDD_out object to be used in SVDD_predict!(). Input is the number of time steps T (in prediction mode). If T for prediction differs from T of the training data (Ttrain) use Ttrain as additional argument.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.KNFST_train",
    "page": "-",
    "title": "MultivariateAnomalies.KNFST_train",
    "category": "Function",
    "text": "KNFST_train(K)\n\ntrain a one class novelty KNFST model on a Kernel matrix K according to Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler: \"Kernel Null Space Methods for Novelty Detection\". Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013.\n\nOutput\n\n(proj, targetValue) proj 	– projection vector for data points (project x via kx*proj, where kx is row vector containing kernel values of x and training data) targetValue – value of all training samples in the null space\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.KNFST_predict",
    "page": "-",
    "title": "MultivariateAnomalies.KNFST_predict",
    "category": "Function",
    "text": "KNFST_predict(model, K)\n\npredict the outlierness of some data (represented by the kernel matrix K), given some KNFST model from KNFST_train(K). Compute Kwith kernel_matrix().\n\nPaul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler: \"Kernel Null Space Methods for Novelty Detection\". Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.KNFST_predict!",
    "page": "-",
    "title": "MultivariateAnomalies.KNFST_predict!",
    "category": "Function",
    "text": "KNFST_predict!(KNFST_out, KNFST_mod, K)\n\npredict the outlierness of some data (represented by the kernel matrix K), given a KNFST_out object (init_KNFST()), some KNFST model (KNFST_mod = KNFST_train(K)) and the testing kernel matrix K.\n\nPaul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler: \"Kernel Null Space Methods for Novelty Detection\". Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013.\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#MultivariateAnomalies.init_KNFST",
    "page": "-",
    "title": "MultivariateAnomalies.init_KNFST",
    "category": "Function",
    "text": "init_KNFST(T, KNFST_mod)\n\ninitialize a KNFST_outobject for the use with KNFST_predict!, given T, the number of observations and the model output KNFST_train(K).\n\n\n\n"
},

{
    "location": "man/DetectionAlgorithms.html#Functions-1",
    "page": "-",
    "title": "Functions",
    "category": "section",
    "text": "REC\nREC!\ninit_REC\nKDE\nKDE!\ninit_KDE\nT2\nT2!\ninit_T2\nKNN_Gamma\nKNN_Gamma!\ninit_KNN_Gamma\nKNN_Delta\nKNN_Delta!\ninit_KNN_Delta\nUNIV\nUNIV!\ninit_UNIV\nSVDD_train\nSVDD_predict\nSVDD_predict!\ninit_SVDD_predict\nKNFST_train\nKNFST_predict\nKNFST_predict!\ninit_KNFST    "
},

{
    "location": "man/DetectionAlgorithms.html#Index-1",
    "page": "-",
    "title": "Index",
    "category": "section",
    "text": ""
},

{
    "location": "man/AUC.html#",
    "page": "-",
    "title": "-",
    "category": "page",
    "text": ""
},

{
    "location": "man/AUC.html#AUC-1",
    "page": "-",
    "title": "AUC",
    "category": "section",
    "text": "Compute true positive rates, false positive rates and the area under the curve to evaulate the algorihtms performance. Efficient implementation according toFawcett, T. (2006). An introduction to ROC analysis. Pattern Recognition Letters, 27(8), 861–874. http://doi.org/10.1016/j.patrec.2005.10.010"
},

{
    "location": "man/AUC.html#MultivariateAnomalies.auc",
    "page": "-",
    "title": "MultivariateAnomalies.auc",
    "category": "Function",
    "text": "auc(scores, events, increasing = true)\n\ncompute the Area Under the receiver operator Curve (AUC), given some output scores array and some ground truth (events). By default, it is assumed, that the scores are ordered increasingly (increasing = true), i.e. high scores represent events.\n\nExamples\n\njulia> scores = rand(10, 2)\njulia> events = rand(0:1, 10, 2)\njulia> auc(scores, events)\njulia> auc(scores, boolevents(events))\n\n\n\n"
},

{
    "location": "man/AUC.html#MultivariateAnomalies.auc_fpr_tpr",
    "page": "-",
    "title": "MultivariateAnomalies.auc_fpr_tpr",
    "category": "Function",
    "text": "auc_fpr_tpr(scores, events, quant = 0.9, increasing = true)\n\nSimilar like auc(), but return additionally the true positive and false positive rate at a given quantile (default: quant = 0.9).\n\nExamples\n\njulia> scores = rand(10, 2)\njulia> events = rand(0:1, 10, 2)\njulia> auc_fpr_tpr(scores, events, 0.8)\n\n\n\n"
},

{
    "location": "man/AUC.html#MultivariateAnomalies.boolevents",
    "page": "-",
    "title": "MultivariateAnomalies.boolevents",
    "category": "Function",
    "text": "boolevents(events)\n\nconvert an events array into a boolean array.\n\n\n\n"
},

{
    "location": "man/AUC.html#Functions-1",
    "page": "-",
    "title": "Functions",
    "category": "section",
    "text": "auc\nauc_fpr_tpr\nboolevents"
},

{
    "location": "man/DistDensity.html#",
    "page": "-",
    "title": "-",
    "category": "page",
    "text": ""
},

{
    "location": "man/DistDensity.html#Distance,-Kernel-Matrices-and-k-Nearest-Neighbours-1",
    "page": "-",
    "title": "Distance, Kernel Matrices and k-Nearest Neighbours",
    "category": "section",
    "text": "Compute distance matrices (similarity matrices), convert them into kernel matrices or k-nearest neighbor objects."
},

{
    "location": "man/DistDensity.html#MultivariateAnomalies.dist_matrix",
    "page": "-",
    "title": "MultivariateAnomalies.dist_matrix",
    "category": "Function",
    "text": "dist_matrix{tp, N}(data::AbstractArray{tp, N}; dist::ASCIIString = \"Euclidean\", space::Int = 0, lat::Int = 0, lon::Int = 0, Q = 0)\ndist_matrix{tp, N}(data::AbstractArray{tp, N}, training_data; dist::ASCIIString = \"Euclidean\", space::Int = 0, lat::Int = 0, lon::Int = 0, Q = 0)\n\ncompute the distance matrix of data or the distance matrix between data and training data i.e. the pairwise distances along the first dimension of data, using the last dimension as variables. dist is a distance metric, currently Euclidean(default), SqEuclidean, Chebyshev, Cityblock, JSDivergence, Mahalanobis and SqMahalanobis are supported. The latter two need a covariance matrix Q as input argument.\n\nExamples\n\njulia> dc = randn(10, 4,3)\njulia> D = dist_matrix(dc, space = 2)\n\n\n\n"
},

{
    "location": "man/DistDensity.html#MultivariateAnomalies.dist_matrix!",
    "page": "-",
    "title": "MultivariateAnomalies.dist_matrix!",
    "category": "Function",
    "text": "dist_matrix!(D_out, data, ...)\n\ncompute the distance matrix of data, similar to dist_matrix(). D_out object has to be preallocated, i.e. with init_dist_matrix.\n\njulia> dc = randn(10,4, 4,3)\njulia> D_out = init_dist_matrix(dc)\njulia> dist_matrix!(D_out, dc, lat = 2, lon = 2)\njulia> D_out[1]\n\n\n\n"
},

{
    "location": "man/DistDensity.html#MultivariateAnomalies.init_dist_matrix",
    "page": "-",
    "title": "MultivariateAnomalies.init_dist_matrix",
    "category": "Function",
    "text": "init_dist_matrix(data)\ninit_dist_matrix(data, training_data)\n\ninitialize a D_out object for dist_matrix!().\n\n\n\n"
},

{
    "location": "man/DistDensity.html#MultivariateAnomalies.knn_dists",
    "page": "-",
    "title": "MultivariateAnomalies.knn_dists",
    "category": "Function",
    "text": "knn_dists(D, k::Int, temp_excl::Int = 5)\n\nreturns the k-nearest neighbors of a distance matrix D. Excludes temp_excl (default: temp_excl = 5) distances from the main diagonal of D to be also nearest neighbors.\n\njulia> dc = randn(20, 4,3)\njulia> D = dist_matrix(dc, space = 2)\njulia> knn_dists_out = knn_dists(D, 3, 1)\njulia> knn_dists_out[5] # distances\njulia> knn_dists_out[4] # indices\n\n\n\n"
},

{
    "location": "man/DistDensity.html#MultivariateAnomalies.knn_dists!",
    "page": "-",
    "title": "MultivariateAnomalies.knn_dists!",
    "category": "Function",
    "text": "knn_dists!(knn_dists_out, D, temp_excl::Int = 5)\n\nreturns the k-nearest neighbors of a distance matrix D. Similar to knn_dists(), but uses preallocated input object knn_dists_out, initialized with init_knn_dists(). Please note that the number of nearest neighbors k is not necessary, as it is already determined by the knn_dists_out object.\n\njulia> dc = randn(20, 4,3)\njulia> D = dist_matrix(dc, space = 2)\njulia> knn_dists_out = init_knn_dists(dc, 3)\njulia> knn_dists!(knn_dists_out, D)\njulia> knn_dists_out[5] # distances\njulia> knn_dists_out[4] # indices\n\n\n\n"
},

{
    "location": "man/DistDensity.html#MultivariateAnomalies.init_knn_dists",
    "page": "-",
    "title": "MultivariateAnomalies.init_knn_dists",
    "category": "Function",
    "text": "init_knn_dists(T::Int, k::Int)\ninit_knn_dists(datacube::AbstractArray, k::Int)\n\ninitialize a preallocated knn_dists_out object. kis the number of nerarest neighbors, T the number of time steps (i.e. size of the first dimension) or a multidimensional datacube.\n\n\n\n"
},

{
    "location": "man/DistDensity.html#MultivariateAnomalies.kernel_matrix",
    "page": "-",
    "title": "MultivariateAnomalies.kernel_matrix",
    "category": "Function",
    "text": "kernel_matrix(D::AbstractArray, σ::Float64 = 1.0[, kernel::ASCIIString = \"gauss\", dimension::Int64 = 1])\n\ncompute a kernel matrix out of distance matrix D, given σ. Optionally normalized by the dimension, if kernel = \"normalized_gauss\". compute D with dist_matrix().\n\njulia> dc = randn(20, 4,3)\njulia> D = dist_matrix(dc, space = 2)\njulia> K = kernel_matrix(D, 2.0)\n\n\n\n"
},

{
    "location": "man/DistDensity.html#MultivariateAnomalies.kernel_matrix!",
    "page": "-",
    "title": "MultivariateAnomalies.kernel_matrix!",
    "category": "Function",
    "text": "kernel_matrix!(K, D::AbstractArray, σ::Float64 = 1.0[, kernel::ASCIIString = \"gauss\", dimension::Int64 = 1])\n\ncompute a kernel matrix out of distance matrix D. Similar to kernel_matrix(), but with preallocated Array K (K = similar(D)) for output.\n\njulia> dc = randn(20, 4,3)\njulia> D = dist_matrix(dc, space = 2)\njulia> kernel_matrix!(D, D, 2.0) # overwrites distance matrix\n\n\n\n"
},

{
    "location": "man/DistDensity.html#Functions-1",
    "page": "-",
    "title": "Functions",
    "category": "section",
    "text": "dist_matrix\ndist_matrix!\ninit_dist_matrix\nknn_dists\nknn_dists!\ninit_knn_dists\nkernel_matrix\nkernel_matrix!"
},

{
    "location": "man/DistDensity.html#Index-1",
    "page": "-",
    "title": "Index",
    "category": "section",
    "text": ""
},

{
    "location": "man/Scores.html#",
    "page": "-",
    "title": "-",
    "category": "page",
    "text": ""
},

{
    "location": "man/Scores.html#Scores-1",
    "page": "-",
    "title": "Scores",
    "category": "section",
    "text": "Postprocess your anomaly scores by making different algorithms comparable and computing their ensemble."
},

{
    "location": "man/Scores.html#MultivariateAnomalies.get_quantile_scores",
    "page": "-",
    "title": "MultivariateAnomalies.get_quantile_scores",
    "category": "Function",
    "text": "get_quantile_scores(scores, quantiles = 0.0:0.01:1.0)\n\nreturn the quantiles of the given N dimensional anomaly scores cube. quantiles (default: quantiles = 0.0:0.01:1.0) is a Float range of quantiles. Any score being greater or equal quantiles[i] and beeing smaller than quantiles[i+1] is assigned to the respective quantile quantiles[i].\n\nExamples\n\njulia> scores1 = rand(10, 2)\njulia> quantile_scores1 = get_quantile_scores(scores1)\n\n\n\n"
},

{
    "location": "man/Scores.html#MultivariateAnomalies.get_quantile_scores!",
    "page": "-",
    "title": "MultivariateAnomalies.get_quantile_scores!",
    "category": "Function",
    "text": "get_quantile_scores!{tp,N}(quantile_scores::AbstractArray{Float64, N}, scores::AbstractArray{tp,N}, quantiles::FloatRange{Float64} = 0.0:0.01:1.0)\n\nreturn the quantiles of the given N dimensional scores array into a preallocated quantile_scores array, see get_quantile_scores().\n\n\n\n"
},

{
    "location": "man/Scores.html#MultivariateAnomalies.compute_ensemble",
    "page": "-",
    "title": "MultivariateAnomalies.compute_ensemble",
    "category": "Function",
    "text": "compute_ensemble(m1_scores, m2_scores[, m3_scores, m4_scores]; ensemble = \"mean\")\n\ncompute the mean (ensemble = \"mean\"), minimum (ensemble = \"min\"), maximum (ensemble = \"max\") or median (ensemble = \"median\") of the given anomaly scores. Supports between 2 and 4 scores input arrays (m1_scores, ..., m4_scores). The scores of the different anomaly detection algorithms should be somehow comparable, e.g., by using get_quantile_scores() before.\n\nExamples\n\njulia> scores1 = rand(10, 2)\njulia> scores2 = rand(10, 2)\njulia> quantile_scores1 = get_quantile_scores(scores1)\njulia> quantile_scores2 = get_quantile_scores(scores2)\njulia> compute_ensemble(quantile_scores1, quantile_scores2, ensemble = \"max\")\n\n\n\n"
},

{
    "location": "man/Scores.html#Functions-1",
    "page": "-",
    "title": "Functions",
    "category": "section",
    "text": "get_quantile_scores\nget_quantile_scores!\ncompute_ensemble"
},

{
    "location": "man/Scores.html#Index-1",
    "page": "-",
    "title": "Index",
    "category": "section",
    "text": ""
},

]}
