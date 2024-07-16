import anndata as ad
import numpy as np
import pandas as pd
import sklearn
from  sklearn.utils.extmath import randomized_svd, squared_norm,safe_sparse_dot
from sklearn.utils import check_random_state, check_array
import time
import scipy 
from scipy import sparse
import sys
import logging
from sklearn.neighbors import kneighbors_graph
import scanpy as sc
import squidpy as sq
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.sparse import csgraph
from numpy.linalg import multi_dot
import warnings

warnings.filterwarnings('ignore')

class intNMF():
	"""
	intNMF
	======
	
	Class to run int NMF on multiome data
	
	Attributes
	---------------
	
	
	Methods
	---------------
	
	""" 

	def __init__(self, adata, n_topics, lam, epochs = 50,init = 'NNDSVD',tf_transf = False,dense=False,random_state = None, mod1_skew = 1):
		"""
		Parameters
		----------
		adata: anndata object with spatial data
		n_topics (k): is the number of latent topics
		epochs: Number of interations during optimisation
		init: initialisation  method. default:SSVD
		random_state: sets seed
		mod1_skew: sets the modality weight with 2 being only gene expression and 0 only image - 1 equally weights both modalities
		Lists to store training metrics.
		"""
		self.data = adata
		self.k = n_topics
		self.epochs = epochs
		self.init = init
		self.dense = dense
		self.loss = []
		self.tf_transf = tf_transf
		self.loss_expr = []
		self.loss_im = []
		self.epoch_times = []
		self.epoch_iter = []
		self.mod1_skew = mod1_skew
		self.random_state = random_state
		self.lam = lam

	
	
	def fit(self, adata,tf_transf = False):
	
		"""optimise NMF. Uses accelerated Hierarchical alternating least squares algorithm proposeed here, but modified to
		joint factorise two matrices. https://arxiv.org/pdf/1107.5194.pdf. Only required arguments are the matrices to use for factorisation.
		GEX and image feature matrices are expected in cell by feature format. Matrices should be scipy sparse matrices.
		min ||X_expr - (theta . phi_expr)||_2 and min ||X_im - (theta . phi_im)||_2 s.t. theta, phi_expr, phi_im > 0. So that theta hold the latent topic scores for a cell. And phi
		the allows recontruction of X
		Parameters
		----------
		expr_mat: scipy matrix of spatial transcriptomics gene expression
		im_mat: scipy matrix of image features
		legacy: Set to False if adata.X.A exists, set to True otherwise"""
		
		start = time.perf_counter()

		if type(adata.X).__name__ in ['csr_matrix', 'SparseCSRView']:
			self.gene_expr = pd.DataFrame(data = adata.X.toarray(), index = adata.obs.index, columns=adata.var_names)
		else:
			self.gene_expr = pd.DataFrame(data = adata.X, index = adata.obs.index, columns=adata.var_names)


		# extract = extract_image_features(adata)

		self.image_features = adata.obsm['features']

		# check for negative image feature values and set to 0
		
		self.image_features[self.image_features < 0]=0


		if tf_transf == True:


			print('Transforming the matrices to TF-IDF ... ')

			expr_mat  = log_tf_idf(self.gene_expr)
			im_mat  = log_tf_idf(self.image_features)

		else:
			expr_mat = self.gene_expr
			im_mat = self.image_features


		# if np.any(expr_mat < 0):
		# 	raise Exception('Some entries of the expression matrix are negative')

		# if np.any(im_mat < 0):
		# 	raise Exception('Some entries of the image features matrix are negative')

		expr_mat = sparse.csr_matrix(expr_mat)
		im_mat = sparse.csr_matrix(im_mat)
			

		print('Fitting Joint NMF ...')

		cells = im_mat.shape[0]
		regions = im_mat.shape[1]
		genes = expr_mat.shape[1]

		EXPR_mat = expr_mat
		IM_mat = im_mat

		# gr = calculate_graph(adata)
		# ds = adata.obsp["spatial_connectivities"].todense()


	   
		# A = kneighbors_graph(adata.X.A, 2, mode='distance', include_self=True)
		# A = A.toarray()
		sc.pp.pca(adata)
		sc.pp.neighbors(adata,knn=False,method='gauss')
		A = adata.obsp['connectivities']
		self.lap = csgraph.laplacian(A)
		
		if tf_transf == True:

			nM_expr = sparse.linalg.norm(EXPR_mat, ord='fro')**2
			nM_im = sparse.linalg.norm(IM_mat, ord='fro')**2
		else:

			nM_expr = sparse.linalg.norm(EXPR_mat, ord='fro')**2
			nM_im = sparse.linalg.norm(IM_mat, ord='fro')**2
			
		#intialise matrices. Default is random. Dense numpy arrays.
		theta, phi_expr, phi_im = self._initialize_nmf(EXPR_mat, IM_mat, self.k, init=self.init,random_state=self.random_state)                

		early_stopper = 0
		counter = 0
		interval = round(self.epochs/10)
		progress = 0
		
		self.loss = []
		
		#perform the optimisation. A, B, theta and phi matrices are modified to fit the update function 
		for i in range(self.epochs):
		
			epoch_start = time.perf_counter()
			#expr updates. 
			#theta

			#update theta
			eit1 = time.perf_counter()
			rnaMHt = safe_sparse_dot(EXPR_mat, phi_expr.T)
			rnaHHt = phi_expr.dot(phi_expr.T)
			imgMHt = safe_sparse_dot(IM_mat, phi_im.T)
			imgHHt = phi_im.dot(phi_im.T)
			eit1 = time.perf_counter() - eit1


			if i == 0:
				scale = ((np.sum(rnaMHt*theta)/np.sum(rnaHHt*(theta.T.dot(theta)))) + 
					np.sum(imgMHt*theta)/np.sum(imgHHt*(theta.T.dot(theta))))/2
				theta=theta*scale

				theta, theta_it = self._HALS_W(theta, rnaHHt, rnaMHt, imgHHt, imgMHt, eit1)

		
			
			#update phi_expr
			eit1 = time.perf_counter()
			A_expr = safe_sparse_dot(theta.T, EXPR_mat)
			B_expr = (theta.T).dot(theta)
			eit1 = time.perf_counter() - eit1
			
			phi_expr, phi_expr_it = self._HALS(phi_expr, B_expr, A_expr, eit1)
			
			
			#image updates
			#update theta

			
			eit1 =time.perf_counter()
			A_im = safe_sparse_dot(theta.T, IM_mat)
			B_im = (theta.T).dot(theta)
			eit1 = time.perf_counter() - eit1
			
			phi_im, phi_im_it = self._HALS(phi_im, B_im, A_im, eit1)

			error_expr = np.sqrt(nM_expr - 2*np.sum(phi_expr*A_expr) + np.sum(B_expr*(phi_expr.dot(phi_expr.T))))
			error_im =  np.sqrt(nM_im - 2*np.sum(phi_im*A_im) + np.sum(B_im*(phi_im.dot(phi_im.T))))
			
			epoch_end =time.perf_counter()
			
			epoch_duration = epoch_end - epoch_start
			logging.info('epoch duration: {}\nloss: {}'.format(epoch_duration, error_expr+error_im))
			logging.info('theta expr iter: {}\nphi expr iter:{}\nphi image iter: {}\n'.format(theta_it, phi_expr_it, phi_im_it))
			self.epoch_times.append(epoch_duration)
			self.loss.append(error_expr+error_im)
			self.loss_expr.append(error_im)
			self.loss_im.append(error_expr)
			self.epoch_iter.append(theta_it + phi_expr_it + phi_im_it)
			#print('total number of iterations in epoch {} is {}'.format(i, theta_rna_it, phi_rna_it + theta_img_it + phi_img_it))
			
			#early stopping condition requires 50 consecutive iterations with no change.
			try:
				if self.loss[-2] < self.loss[-1]:
					early_stopper += 1
				elif early_stopper > 0:
					early_stopper = 0
				if early_stopper > 50:
					break
			except IndexError:
				continue

			counter += 1

		self.theta = theta
		self.phi_expr = pd.DataFrame(phi_expr)
		self.phi_im = pd.DataFrame(phi_im)
		self.error_expr = error_expr
		self.error_im = error_im


		
		end = time.perf_counter()	
		# self.theta = theta
		


		self.total_time = end - start
		logging.info(self.total_time)
		l = []
		for i in range(0,len(self.gene_expr)):
			nor = NormalizeData(theta[i,:])
			l.append(nor)

		theta_nor = pd.DataFrame(l)

		print('Adding cell proportion values to adata.obs...')

		n = theta_nor.shape[1]
		
		for i in range(n):
			adata.obs[f"deconv_{i}"] = theta_nor.iloc[:,i].values


		del EXPR_mat
		del IM_mat

	


	
		
	def _HALS(self, V, UtU, UtM, eit1, alpha=0.5, delta=0.1):
		"""Optimizing min_{V >= 0} ||M-UV||_F^2 with an exact block-coordinate descent schemeUpdate V.
		UtU and UtM are exprensive to compute so multiple updates are calcuated. 
		Parameters:
		-----------
		V: Array like mat to update
		UtU: precomputed dense array
		UtM: precomputed dense array
		eit1: eopch start time
		alpha: control time based stop criteria
		delta: control loss based stop criteria
		Returns
		---------------------
		V: optimised array
		n_it: number of iterations (usually 6/7)"""
		r, n = V.shape
		eit2 = time.perf_counter()
		cnt = 1
		eps = 1
		eps0 = 1
		eit3 = 0
		n_it = 0
		
		while cnt == 1 or ((time.perf_counter()-eit2 < (eit1+eit3)*alpha) and (eps >= (delta**2)*eps0)):
			nodelta=0
			if cnt == 1:
				eit3 = time.perf_counter()
			
		
			for k in range(r):
				deltaV = np.maximum((UtM[k,:]-UtU[k,:].dot(V))/UtU[k, k], -V[k, :])

				V[k,:] = V[k,:] + deltaV
				nodelta = nodelta + deltaV.dot(deltaV.T)
				V[k,V[k,:] == 0] =   1e-16*np.max(V)

			if cnt == 1:
				eps0 = nodelta
				eit3 = time.perf_counter() - eit3
			   
			eps = nodelta
			cnt = 0
			n_it += 1
			
		return V, n_it
		



	def _HALS_W(self, W, rnaHHt, rnaMHt, imgHHt, imgMHt, eit1, alpha=0.5, delta=0.1):
	    """Optimizing min_{V >= 0} ||M-UV||_F^2 with an exact block-coordinate descent schemeUpdate V.
	    UtU and UtM are exprensive to compute so multiple updates are calcuated. 
	    Parameters:
	    -----------
	    W: Array like mat to update
	    imgHHt: precomputed dense array
	    imgMHt: precomputed dense array
	    rnaHHt: precomputed dense array
	    rnaMHt: precomputed dense array
	    eit1: precompute time
	    alpha: control time based stop criteria
	    delta: control loss based stop criteria
	    Returns
	    ---------------------
	    V: optimised array
	    n_it: number of iterations (usually 6/7)
	    
	    stop condition
	    
	    (epoch run time ) < (precompute time + current iteration time)/2
	    AND
	    sum of squares of updates to V at current epoch > 0.01 * sum of squares of updates to V at first epoch
	    """
	    n, K = W.shape
	    eit2 = time.perf_counter()  # start time
	    cnt = 1
	    eps = 1
	    eps0 = 1
	    eit3 = 0  # iteration time
	    n_it = 0
	    lam = self.lam
	    mod1_skew = self.mod1_skew

	    
	    while cnt == 1 or ((time.perf_counter()-eit2 < (eit1+eit3)*alpha) and (eps >= (delta**2)*eps0)):
	        nodelta=0
	        if cnt == 1:
	            eit3 = time.perf_counter()
	        
	    
	        for k in range(K):

	            #print(W.shape, imgHHt.shape)
	            #print(W.dot(imgHHt[:,k]).shape)
	            deltaW = np.maximum(((mod1_skew*(rnaMHt[:,k] -W.dot(rnaHHt[:,k])) + (2-mod1_skew)*(imgMHt[:,k]-W.dot(imgHHt[:,k])))/
                         ((mod1_skew*rnaHHt[k, k]) + ((2-mod1_skew)*imgHHt[k,k])) ), -W[:, k]) 
	            # deltaW = np.maximum(((mod1_skew*(rnaMHt[:,k] -W.dot(rnaHHt[:,k])) + (2-mod1_skew)*(imgMHt[:,k]-W.dot(imgHHt[:,k])))/
             #             (lam*np.dot(self.lap,W[:,k])+(mod1_skew*rnaHHt[k, k]) + ((2-mod1_skew)*imgHHt[k,k])) ), -W[:, k]) 

	            W[:,k] = W[:,k] + deltaW 
	            # lam*np.trace(np.transpose(W)*self.lap*W)
	            nodelta = nodelta + deltaW.dot(deltaW.T)
	            W[W[:,k] == 0, k] =   1e-16*np.max(W)
	            

	        if cnt == 1:
	            eps0 = nodelta
	            eit3 = time.perf_counter() - eit3
	           
	        eps = nodelta
	        cnt = 0
	        n_it += 1
	        
	    return W, n_it


	
	# def _initialise_nmf_im_matrix(self, X, IM_mat, n_components, eps=1e-6, random_state=None,init='NNDSVD',dense=False):

	# 	U,S,V = randomized_svd(IM_mat, n_components, random_state=random_state)
		
	# 		W, H_im = np.zeros(U.shape), np.zeros(V.shape)

	# 		W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])

	# 		H_im[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

	# 		for j in range(1, n_components):
	# 			x, y = U[:, j], V[j, :]

	# 			# extract positive and negative parts of column vectors
	# 			x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
	# 			x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

	# 			# and their norms
	# 			x_p_nrm, y_p_nrm = np.sqrt(squared_norm(x_p)), np.sqrt(squared_norm(y_p))
	# 			x_n_nrm, y_n_nrm = np.sqrt(squared_norm(x_n)), np.sqrt(squared_norm(y_n))

	# 			m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

	# 				# choose update
	# 			if m_p > m_n:
	# 				u = x_p / x_p_nrm
	# 				v = y_p / y_p_nrm
	# 				sigma = m_p
	# 			else:
	# 				u = x_n / x_n_nrm
	# 				v = y_n / y_n_nrm
	# 				sigma = m_n

	# 			lbd = np.sqrt(S[j] * sigma)
	# 			W[:, j] = lbd * u
	# 			H_im[j, :] = lbd * v

	# 		if dense == True:
	# 		# NNDSVDa
	# 			avg = IM_mat.mean()
	# 			W[W == 0] = avg
	# 			H_im[H_im == 0] = avg
	# 			H = safe_sparse_dot(np.linalg.pinv(W), X,dense_output=True)
	# 			H[H < eps] = 0

	# 			H = safe_sparse_dot(np.linalg.pinv(W), X,dense_output=True)
	# 			H[H < eps] = 0

	# 		else:	
	# 			W[W < eps] = 0
	# 			H_im[H_im < eps] = 0
			
	# 			H = safe_sparse_dot(np.linalg.pinv(W), X)
	# 			H[H < eps] = 0

			 
	# 	return H_im

	
	
	### PLAN OT CHANGE THIS TO GILLIS METHOD WITH AUTOMATIC TOPIC DETECTION
	#https://github.com/CostaLab/scopen/blob/6be56fac6470e5b6ecdc5a2def25eb60ed6a1bcc/scopen/MF.py#L696    
	def _initialize_nmf(self, X, IM_mat, n_components, eps=1e-6, random_state=None,init='NNDSVD',dense=False):

			"""Algorithms for NMF initialization.
			Computes an initial guess for the non-negative
			rank k matrix approximation for X: X = WH
			Parameters IM_mat_tfidf_log
			----------
			X : array-like, shape (n_samples, n_features)
				The data matrix to be decomposed.
			n_components : integer
				The number of components desired in the approximation.
			init :  None | 'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar'
				Method used to initialize the procedure.
				Default: None.
				Valid options:
				- None: 'nndsvd' if n_components <= min(n_samples, n_features),
					otherwise 'random'.
				- 'random': non-negative random matrices, scaled with:
					sqrt(X.mean() / n_components)
				- 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
			eps : float
				Truncate all values less then this in output to zero.
			random_state : int, RandomState instance or None, optional, default: None
				If int, random_state is the seed used by the random number generator;
				If RandomState instance, random_state is the random number generator;
				If None, the random number generator is the RandomState instance used
				by `np.random`. Used when ``random`` == 'nndsvdar' or 'random'.
			Returns
			-------
			W : array-like, shape (n_samples, n_components)
				Initial guesses for solving X ~= WH
			H : array-like, shape (n_components, n_features)
				Initial guesses for solving X ~= WH
			H_img : array-like, shape (n_components, n_features)
				Initial guesses for solving X_img ~= WH_img
			References
			----------
			C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
			nonnegative matrix factorization - Pattern Recognition, 2008, 
			http://tinyurl.com/nndsvd"""

			n_samples, n_features = X.shape

			# if init == 'user_input':
			# 	return W_in, H_1_in, H_2_in

			if not sparse.issparse(X) and np.any(np.isnan(X)):
				raise ValueError("NMF initializations with NNDSVD are not available "
								 "with missing values (np.nan).")
								 
			if init == 'random':
				X_mean = X.mean()
				avg = np.sqrt(X_mean / n_components)
				avg_im = np.sqrt(IM_mat.mean() / n_components)
				
				rng = check_random_state(random_state)
				H = avg * rng.randn(n_components, n_features)
				W = avg * rng.randn(n_samples, n_components)
				H_im = avg_im * rng.randn(n_components, IM_mat.shape[1])

				# we do not write np.abs(H, out=H) to stay compatible with
				# numpy 1.5 and earlier where the 'out' keyword is not
				# supported as a kwarg on ufuncs
				np.abs(H, H)
				np.abs(W, W)
				np.abs(H_im, H_im)
				return W, H, H_im

			if init == 'kmeans':

				kmeans = KMeans(n_clusters=n_components, init='k-means++',max_iter=500).fit(self.gene_expr.values)
				W = kmeans.fit_transform(self.gene_expr.values)
				H = kmeans.cluster_centers_


				kmeans_im = KMeans(n_clusters=n_components, init='k-means++',max_iter=500).fit(self.image_features.values)
				H_im = kmeans_im.cluster_centers_
				# H_im[H_im < eps] = 0

				return W,H,H_im
			
			if init == 'fuzzy':

				from fcmeans import FCM
				fcm = FCM(n_clusters=n_components)
				fcm.fit(X)

			# NNDSVD initialization
			# X_mean = X - X.mean(axis=1)
			if self.mod1_skew != 0:

				U, S,V = randomized_svd(X, n_components,random_state=random_state)
		
				W, H = np.zeros(U.shape), np.zeros(V.shape)

			# The leading singular triplet is non-negative
			# so it can be used as is for initialization.
				W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])

				H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

				for j in range(1, n_components):
					x, y = U[:, j], V[j, :]

				# extract positive and negative parts of column vectors
					x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
					x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

				# and their norms
					x_p_nrm, y_p_nrm = np.sqrt(squared_norm(x_p)), np.sqrt(squared_norm(y_p))
					x_n_nrm, y_n_nrm = np.sqrt(squared_norm(x_n)), np.sqrt(squared_norm(y_n))

					m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

					# choose update
					if m_p > m_n:
						u = x_p / x_p_nrm
						v = y_p / y_p_nrm
						sigma = m_p
					else:
						u = x_n / x_n_nrm
						v = y_n / y_n_nrm
						sigma = m_n

					lbd = np.sqrt(S[j] * sigma)
					W[:, j] = lbd * u
					H[j, :] = lbd * v

				if dense == True:
			# NNDSVDa
					avg = X.mean()
					W[W == 0] = avg
					H[H == 0] = avg
					H_im = safe_sparse_dot(np.linalg.pinv(W), IM_mat,dense_output=True)
					H_im[H_im < eps] = 0
				# n1 = len(W[W==0])
				# n2 = len(H[H==0])
				# W[W==0] = avg*np.random.uniform(n1,1)/100
				# H[H==0] = avg*np.random.uniform(n2,1)/100
					H_im = safe_sparse_dot(np.linalg.pinv(W), IM_mat,dense_output=True)
					H_im[H_im < eps] = 0

				else:	
					W[W < eps] = 0
					H[H < eps] = 0	


			
					H_im = safe_sparse_dot(np.linalg.pinv(W), IM_mat)
					H_im[H_im < eps] = 0

			elif self.mod1_skew == 2:

				U, S,V = randomized_svd(X, n_components, random_state=random_state)
		
				W, H = np.zeros(U.shape), np.zeros(V.shape)

			# The leading singular triplet is non-negative
			# so it can be used as is for initialization.
				W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])

				H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

				for j in range(1, n_components):
					x, y = U[:, j], V[j, :]

				# extract positive and negative parts of column vectors
					x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
					x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

				# and their norms
					x_p_nrm, y_p_nrm = np.sqrt(squared_norm(x_p)), np.sqrt(squared_norm(y_p))
					x_n_nrm, y_n_nrm = np.sqrt(squared_norm(x_n)), np.sqrt(squared_norm(y_n))

					m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

					# choose update
					if m_p > m_n:
						u = x_p / x_p_nrm
						v = y_p / y_p_nrm
						sigma = m_p
					else:
						u = x_n / x_n_nrm
						v = y_n / y_n_nrm
						sigma = m_n

					lbd = np.sqrt(S[j] * sigma)
					W[:, j] = lbd * u
					H[j, :] = lbd * v

				if dense == True:
			# NNDSVDa
					avg = X.mean()
					W[W == 0] = avg
					H[H == 0] = avg
					H_im = safe_sparse_dot(np.linalg.pinv(W), IM_mat,dense_output=True)
					H_im[H_im < eps] = 0
				# n1 = len(W[W==0])
				# n2 = len(H[H==0])
				# W[W==0] = avg*np.random.uniform(n1,1)/100
				# H[H==0] = avg*np.random.uniform(n2,1)/100
					H_im = 0
					H_im[H_im < eps] = 0

				else:	
					W[W < eps] = 0
					H[H < eps] = 0	
					H_im = 0
					H_im[H_im < eps] = 0
			else:
				U,S,V = randomized_svd(IM_mat, n_components, random_state=random_state)
		
				W, H_im = np.zeros(U.shape), np.zeros(V.shape)

				W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])

				H_im[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

				for j in range(1, n_components):
					x, y = U[:, j], V[j, :]

				# extract positive and negative parts of column vectors
					x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
					x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

				# and their norms
					x_p_nrm, y_p_nrm = np.sqrt(squared_norm(x_p)), np.sqrt(squared_norm(y_p))
					x_n_nrm, y_n_nrm = np.sqrt(squared_norm(x_n)), np.sqrt(squared_norm(y_n))

					m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

					# choose update
					if m_p > m_n:
						u = x_p / x_p_nrm
						v = y_p / y_p_nrm
						sigma = m_p
					else:
						u = x_n / x_n_nrm
						v = y_n / y_n_nrm
						sigma = m_n

					lbd = np.sqrt(S[j] * sigma)
					W[:, j] = lbd * u
					H_im[j, :] = lbd * v

				if dense == True:
			# NNDSVDa
					avg = IM_mat.mean()
					W[W == 0] = avg
					H_im[H_im == 0] = avg
					H = safe_sparse_dot(np.linalg.pinv(W), X,dense_output=True)
					H[H < eps] = 0

					H = safe_sparse_dot(np.linalg.pinv(W), X,dense_output=True)
					H[H < eps] = 0

				else:	
					W[W < eps] = 0
					H_im[H_im < eps] = 0
			
					H = safe_sparse_dot(np.linalg.pinv(W), X)
					H[H < eps] = 0

			 
			return W, H, H_im


def calculate_graph(adata):
	from scipy.sparse import csgraph

	G = sq.gr.spatial_neighbors(adata, n_rings=2,  coord_type= 'Grid', n_neighs=6)





def extract_image_features(adata, u_img = False, img = None, layer = None, 
						library_id = None, features = 'histogram',
						mask_circle = False,
						scale_img = 1,
						spot_scale = 2.5, 
						key_added="features", show_progress_bar = True, n_jobs = None):

		"""Extract image features using squidpy.im.calculate_image_features function from Squidpy https://squidpy.readthedocs.io/en/stable/#
		:adata: anndata object with spatial data
		:layer: mage layer in img that should be processed. If None and only 1 layer is present, it will be selected.
		:library_id: if None, there should only exist one entry in anndata.AnnData.uns ['{spatial_key}'].
		If a str, first search anndata.AnnData.obs ['{library_id}'] which contains the mapping from observations to library ids, then search anndata.AnnData.uns ['{spatial_key}'].
		:features(str: Features to be calculated. Valid options are: 'texture', 'summary','histogram','segmentation','custom'
		:key_added(str):  Key in anndata.AnnData.obsm where to store the calculated features 
		:n_jobs(optional,int): Number of parallel jobs
		:show_progress_bar:– Whether to show the progress bar or not  
		"""
		from PIL import Image
  		
		library_id = list(adata.uns['spatial'].keys())[0]

		if  u_img == True:
			img = img
		else:
			# img: High-resolution image
			img = sq.im.ImageContainer(
				adata.uns["spatial"][library_id]["images"]["hires"],
				scale=adata.uns["spatial"][library_id]["scalefactors"]["tissue_hires_scalef"]
			)
		print('Extracting Image Features ...')
		
		features = sq.im.calculate_image_features(adata, img, features=features, mask_circle=mask_circle,scale = scale_img, spot_scale = spot_scale, key_added="features", show_progress_bar=True)
		
		adata.obsm[key_added].columns = ad.utils.make_index_unique(adata.obsm[key_added].columns)


# def model_selection(adata, n_components_start, n_components_end):

# 	""" Model selection based on elbow method: Calculates the within-cluster-sum of squared errors (wss) for different values of n_components and chooses the n_components 
# 	for which WSS starts to level off
# 	:adata: anndata object with spatial data
# 	:n_components_start: number of topics start range 
# 	:n_components_end:  number of topics end range

# 	"""

# 	wss= []
# 	for i in range(n_components_start,n_components_end):

# 		kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300, n_init=10,random_state=0)
# 		kmeans.fit(adata.X.A)
# 		wss.append(kmeans.inertia_)

# 	plt.plot(range(n_components_start,n_components_end),wss)
# 	plt.title('Model Selection Elbow')
# 	plt.xlabel('Number of Topics')
# 	plt.ylabel('Inertia')
# 	plt.show()


def rrssq(adata,theta,phi_expr):

	norm = sc.pp.normalize_total(adata)
	actual = adata.X.A

	pred = np.matmul(theta,phi_expr)


def model_selection(adata,n_components_start,n_components_end,niter=30):

	""" Model selection based on elbow method: Calculates the within-cluster-sum of squared errors (wss) for different values of n_components and chooses the n_components 
	for which WSS starts to level off
	:adata: anndata object with spatial data
	:n_components_start: number of topics start range 
	:n_components_end:  number of topics end range

	"""
	# sc.pp.normalize_total(adata)
	# extract_image_features(adata,
 #    scale_img = 0.2,
 #    spot_scale = 4)

	# for i in range(0,niter):

	# 	intNMF(adata,epochs = 50, init = 'random',mod1_skew=1)
	# 	nmf_model.fit(adata)

	# 	_, idx = argmax(self.H, axis=0)
 #        mat1 = repmat(idx, self.V.shape[1], 1)
 #        mat2 = repmat(idx.T, 1, self.V.shape[1])
 #        cons = elop(mat1, mat2, eq)
 #        if not hasattr(self, 'consold'):
 #            self.cons = cons
 #            self.consold = np.mat(np.logical_not(cons))
 #        else:
 #            self.consold = self.cons
 #            self.cons = cons
 #        conn_change = elop(self.cons, self.consold, ne).sum()
 #        return conn_change > 0


	wss= []
	for i in range(n_components_start,n_components_end):

		kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300, n_init=10,random_state=0)
		kmeans.fit(adata.X.A)
		wss.append(kmeans.inertia_)


	plt.plot(range(n_components_start,n_components_end),wss)
	plt.title('Model Selection Elbow')
	plt.xlabel('Number of Topics')
	plt.ylabel('Inertia')
	plt.show()


	


def log_tf_idf(mat_in, scale = 10000):
	"""
	Return a TF-IDF transformed matrix.
   
   :param mat_in: Positional (required) matrix to be TF-IDF transformed. Should be cell x feature. 
   :type mat_in: `csr matrix type <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`_, or other sparse/dense matrices which can be coerced to csr format. 
   :raise nmf_models.InvalidMatType: If the mat type is invalid
   :param scale: Optional number to scale values by.
   :type scale: int
   :return: TF-IDF transformed matrix
   :rtype: csr matrix
	"""
	mat = sparse.csr_matrix(mat_in)
	
	cell_counts = mat.sum(axis = 1)
	for row in range(len(cell_counts)):
		mat.data[mat.indptr[row]:mat.indptr[row+1]] = (mat.data[mat.indptr[row]:mat.indptr[row+1]]*scale)/cell_counts[row]

	mat = sparse.csc_matrix(mat)
	[rows, cols] = mat.nonzero()
	feature_count = np.zeros(mat.shape[1])
	unique, counts = np.unique(cols, return_counts=True)

	feature_count[unique] = counts
	for col in range(len(feature_count)):
		mat.data[mat.indptr[col]:mat.indptr[col+1]] = mat.data[mat.indptr[col]:mat.indptr[col+1]]*(mat.shape[0]/(feature_count[col]+1))  #idf Number of cells/Number of cells region is open in + 1
	mat = mat.log1p()
	return mat

def NormalizeData(data):

	return (data / data.sum())


def plot_topic_proportions(adata,topics):

	from re import sub # create nice names

	sel_clust = ['deconv_'+str(i) for i in range(topics)]
	with mpl.rc_context({'figure.figsize': (5, 6), 'axes.facecolor': 'black'}):
		sc.pl.spatial(adata,
				cmap='magma',
				color=sel_clust,
				ncols=5,
				size=1, img_key='hires', 
				alpha_img=0.4,
				vmin=0
				 )


def get_genes_topic(adata,phi_expr):

	genes = phi_expr

	genes.columns=adata.var_names

	genes = genes.T

	topic_genes = genes.sort_values(by=[0],ascending=False)
	topic_genes = topic_genes.T

	return topic_genes

class InvalidMatType(Exception):
	"""Raised if the mat is invalid."""
	pass
	

