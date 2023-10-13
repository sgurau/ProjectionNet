# 12 Oct 2023
1. CV on SVM with precomputed kernel for both $C$ and $\gamma$, starting with $\gamma=30,000$ and $C=1$.
2. Try
2.1 MSnet: https://arxiv.org/pdf/2201.10145.pdf (code: https://github.com/GitZH-Chen/MSNet/tree/main in matlab)
2.2 ManifoldNet: https://arxiv.org/pdf/1809.06211.pdf (code: https://github.com/jjbouza/manifold-net-vision in python)
   for prediction of new observations.


# 5 Oct 2023
1. Understand the problem of SVM classification for new subjects using geodesic distance via MDS. The reason is MDS results may be traslated and rotated each time it runs so the results won't be the same. See 28/9/2023 notes. So we need to find the translation (a vector) and rotation (unitary matrix) to align $\mathbf X_{-s}$ to $\mathbf X$.
2. Apply KNN now with geodesic distance and do CV on $k$. 
3. Create thesis draft to include all the results so that future results collection work can be minised.
4. Think of any method that solve the prediction problem for future data. SVM with precomputed kernel (using geogesic distance) is a potentical cnadidate or dimension reduction methods that learn a projection to replace MDS will be also applicable. 

# 28 Sep 2023
1. Refine hyperparameters for SVM with RBF kernel using MDS 2D resutls by refining the search grid.
2. Make item 1 as standard procedure in future work.
3. Apply SVM with RBF kernel using MDS 3D, 4D etc resutls with hyperparameters fine tuning.
4. Think about the problem for classifying new observations. First step: to see the difference between results of MDS using original distance matrix and MDS using distance matrix with one observation removed. See the following explanation in math.

Let $\mathbf D$ be the distance matrix of size $n\times n$, ie  $\mathbf D\in \mathbb R^{n\times n}$. Write $\mathbf D_{-s}$ as the matrix with columns and rows removed in $\mathbf D$ where $s$ is a set with indices of observations, say $s=\\{1,2\\}$. Then $\mathbf D_{-86}$ would be the distance matrix with the 86th observation removed, i.e. a matrix of size $85\times 85$. Write $$\mathbf X = MDS(\mathbf D,2)$$ as the results of MDS with distance matrix $\mathbf D$ in dim-2, ie 2D space, and $\mathbf X\in \mathbb R^{n\times 2}$. So the claim is $$\mathbf X_{-s} \not = MDS(\mathbf D_{-s},2)$$ where $\mathbf X_{-s}$ is the result of $\mathbf X$ with rows indexed by $s$ removed. 

   
# 21 Sep 2023
1. Push all code and data into github.
2. Apply MDS with Euclidean distance and compare all dimensionality reduction results in thesis
3. Fix SVM code using precomputed kernel
4. Apply SVM with RBF kernel on MDS + SPD geodesic distance matrix results (2D) with cross validation 

   
# 8 June 2023
1. Download and process data for analysis.
2. Read Lama2021 paper and replicate their methods on the data.
3. Sort out HPC connection if possible. 

# 1 June 2023
1. batch processing script for all the collected data for replicability purpose. 

# 18 May 2023
TWo directions: 
1. Data analysis on fMRI -> brain network, e.g. new classification models, the effects of parameters in data acquisition to down-stream analysis, etc. 
2. neuroscience questions we can answer using ADNI or other data sets (need Gen's input)

Things to do 
1. Apply AAL atlas 
2. Investigate slice time correction (whether ADNI data need it or not) 
3. Visualise brain networks (in matlab, not urgent) 

# 19 April 2023

1. complete the preprossessing of fMRI images to obtain brains and active regions 
2. implement any brain atlas to extract nodes and superimpose to the brain activities data to form brain networks 
