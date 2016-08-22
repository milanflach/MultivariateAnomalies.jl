# Learning method for novelty detection with KNFST according to the work:
#
# Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler:
# "Kernel Null Space Methods for Novelty Detection".
# Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013.
#
# Please cite that paper if you are using this code!
#
# proj = calculateKNFST(K, labels)
#
# calculates projection matrix of KNFST
#
# INPUT:
#   K -- (n x n) kernel matrix containing similarities of n training samples
#   labels -- (n x 1) vector containing (multi-class) labels of the n training samples
#
# OUTPUT:
#   proj -- projection matrix for data points (project x via kx*proj,
#           where kx is row vector containing kernel values of x and
#           training data)
#
# (LGPL) copyright by Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler
#
function calculateKNFST(K, labels)

    classes = unique(labels);

    ### check labels
    if length(unique(labels)) == 1
      error("calculateKNFST.jl: not able to calculate a nullspace from data of a single class using KNFST (input variable 'labels' only contains a single value)");
    end

    ### check kernel matrix
    (n,m) = size(K);
    if n != m
        error("calculateKNFST.jl: kernel matrix must be quadratic");
    end

    ### calculate weights of orthonormal basis in kernel space
    centeredK = copy(K); # because we need original K later on again
    centerKernelMatrix(centeredK);
    (basisvecsValues,basisvecs) = eig(centeredK);
    basisvecs = basisvecs[:,basisvecsValues .> 1e-12];
    basisvecsValues = basisvecsValues[basisvecsValues .> 1e-12];
    basisvecsValues = diagm(1./sqrt(basisvecsValues));
    basisvecs = basisvecs*basisvecsValues;

    ### calculate transformation T of within class scatter Sw:
    ### T= B'*Sw*B = H*H'  and H = B'*K*(I-L) and L a block matrix
    L = zeros(n,n);
    for i=1:length(classes)

       L[labels.==classes[i],labels.==classes[i]] = 1./sum(labels.==classes[i]);

    end

    ### need Matrix M with all entries 1/m to modify basisvecs which allows usage of
    ### uncentered kernel values:  (eye(size(M))-M)*basisvecs
    M = ones(m,m)./m;

    ### compute helper matrix H
    H = ((eye(m)-M)*basisvecs)'*K*(eye(n)-L);

    ### T = H*H' = B'*Sw*B with B=basisvecs
    T = H*H';

    ### calculate weights for null space
    eigenvecs = nullspace(T);

    if size(eigenvecs,2) < 1

      (eigenvals,eigenvecs) = eig(T);
      (min_val,min_ID) = findmin(eigenvals);
      eigenvecs = eigenvecs[:,min_ID];

    end

    ### calculate null space projection and return it
    proj = ((eye(m)-M)*basisvecs)*eigenvecs;

end

#############################################################################################################

function centerKernelMatrix(kernelMatrix)
# centering the data in the feature space only using the (uncentered) Kernel-Matrix
#
# INPUT:
#       kernelMatrix -- uncentered kernel matrix
# OUTPUT:
#       centeredKernelMatrix -- centered kernel matrix

  ### get size of kernelMatrix
  n = size(kernelMatrix, 1);

  ### get mean values of each row/column
  columnMeans = mean(kernelMatrix,1); ### NOTE: columnMeans = rowMeans because kernelMatrix is symmetric
  matrixMean = mean(columnMeans);

  centeredKernelMatrix = kernelMatrix;

  for k=1:n

    centeredKernelMatrix[k,:] = centeredKernelMatrix[k,:] - columnMeans;
    centeredKernelMatrix[:,k] = centeredKernelMatrix[:,k] - columnMeans';

  end

  centeredKernelMatrix = centeredKernelMatrix + matrixMean;

end


# Learning method for one-class classification with KNFST according to the work:
#
# Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler:
# "Kernel Null Space Methods for Novelty Detection".
# Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013.
#
# Please cite that paper if you are using this code!
#
#
# function (proj,targetValue) = learn_oneClassNovelty_knfst(K)
#
# compute one-class KNFST model by separating target data from origin in feature space
#
# INPUT:
#   K -- (n x n) kernel matrix containing pairwise similarities between n training samples
#
# OUTPUTS:
#   proj 	-- projection vector for data points (project x via kx*proj,
#           	   where kx is row vector containing kernel values of x and
#           	   training data)
#   targetValue -- value of all training samples in the null space
#
# (LGPL) copyright by Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler
#
function KNFST_train(K)

    # get number of training samples
    n = size(K,1);

    # include dot products of training samples and the origin in feature space (these dot products are always zero!)
    K_ext = [K zeros(n,1); zeros(1,n) 0];

    # create one-class labels + a different label for the origin
    labels = push!(ones(n),0);

    # get model parameters
    proj = calculateKNFST(K_ext,labels);
    targetValue = mean(K_ext[labels.==1,:]*proj,1);
    proj = proj[1:n,:];

    # return both variables
    return proj, targetValue

end