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
from scipy.sparse.linalg import svds
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

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_rel

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

	def __init__(self, adata, n_topics, reg = None,epochs = 50, lam=0,init = 'NNDSVD',l1_weight=1e-4, tol=1e-4,dense=False,random_state = 123, mod1_skew = 1,max_time=None):
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
		self.reg = reg
		self.loss = []
		self.loss_expr = []
		if self.reg:
			self.l1_weight = l1_weight
		self.loss_im = []
		self.epoch_times = []
		self.epoch_iter = []
		self.mod1_skew = mod1_skew
		self.random_state = random_state
		self.lam = lam
		self.max_time = max_time
		self.tol = tol

	def get_params(self, deep=True):
		return {
			'adata': self.adata,
			'n_topics': self.n_topics,
			'epochs': self.epochs,
			'init': self.init,
			'dense': self.dense,
			'tf_transf': self.tf_transf,
			'mod1_skew': self.mod1_skew,
			'random_state': self.random_state,
			'lam': self.lam
		}

	def set_params(self, **params):
		for param, value in params.items():
			setattr(self, param, value)
		return self

	
	
	def fit(self, adata, fixed_H_expr=None, fixed_H_im=None, fixed_W = None):
	
		"""optimise NMF. Uses accelerated Hierarchical alternating least squares algorithm proposeed here, but modified to
		joint factorise two matrices. https://arxiv.org/pdf/1107.5194.pdf. Only required arguments are the matrices to use for factorisation.
		GEX and image feature matrices are expected in cell by feature format. Matrices should be scipy sparse matrices.
		min ||X_expr - (theta . phi_expr)||_2 and min ||X_im - (theta . phi_im)||_2 s.t. theta, phi_expr, phi_im > 0. So that theta hold the latent topic scores for a cell. And phi
		the allows recontruction of X
		Parameters
		----------
		expr_mat: scipy matrix of spatial transcriptomics gene expression
		im_mat: scipy matrix of image features
		"""
		np.random.default_rng(self.random_state)
		start = time.perf_counter()

		if type(adata.X).__name__ in ['csr_matrix', 'SparseCSRView']:
			self.gene_expr = pd.DataFrame(data = adata.X.toarray(), index = adata.obs.index, columns=adata.var_names)

		else:
			self.gene_expr = pd.DataFrame(data = adata.X, index = adata.obs.index, columns=adata.var_names)

		self.image_features = adata.obsm['features']

		# check for negative image feature values and set to 0
		
		self.image_features[self.image_features < 0]=0


		expr_mat = self.gene_expr
		im_mat = self.image_features


		# if np.any(expr_mat < 0):
		# 	raise Exception('Some entries of the expression matrix are negative')

		# if np.any(im_mat < 0):
		# 	raise Exception('Some entries of the image features matrix are negative')

		expr_mat = sparse.csr_matrix(expr_mat)
		im_mat = sparse.csr_matrix(im_mat)
			

		print('Fitting CellPie ...')



		cells = im_mat.shape[0]
		regions = im_mat.shape[1]
		genes = expr_mat.shape[1]

		EXPR_mat = expr_mat
		IM_mat = im_mat
		


		nM_expr = sparse.linalg.norm(EXPR_mat, ord='fro')**2
		nM_im = sparse.linalg.norm(IM_mat, ord='fro')**2
			

		self.theta, self.phi_expr, self.phi_im = self._initialize_nmf(EXPR_mat, IM_mat, self.k, init=self.init)

		if fixed_H_im is not None:
			self.phi_im = pd.DataFrame(fixed_H_im)             

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
			rnaMHt = safe_sparse_dot(EXPR_mat, self.phi_expr.T)
			rnaHHt = self.phi_expr.dot(self.phi_expr.T)
			imgMHt = safe_sparse_dot(IM_mat, self.phi_im.T)
			imgHHt = self.phi_im.dot(self.phi_im.T)
			eit1 = time.perf_counter() - eit1

			if i == 0:
				if fixed_W is None:
					scale = ((np.sum(rnaMHt * self.theta) / np.sum(rnaHHt * (self.theta.T.dot(self.theta)))) + np.sum(imgMHt * self.theta) / np.sum(imgHHt * (self.theta.T.dot(self.theta)))) / 2
					self.theta = self.theta * scale
					self.theta, theta_it = self._HALS_W(self.theta, rnaHHt, rnaMHt, imgHHt, imgMHt, eit1)
				else:
					self.theta = fixed_W
					theta_it = 0
			
			#update phi_expr
			eit1 = time.perf_counter()
			A_expr = safe_sparse_dot(self.theta.T, EXPR_mat)
			B_expr = (self.theta.T).dot(self.theta)
			eit1 = time.perf_counter() - eit1
			
			if fixed_H_expr is not None:
				self.phi_expr = fixed_H_expr
				self.phi_expr_it = 0				
			else:
				self.phi_expr, self.phi_expr_it = self._HALS(self.phi_expr, B_expr, A_expr,eit1)
				
			
			
			#image updates
			#update theta

			
			eit1 =time.perf_counter()
			A_im = safe_sparse_dot(self.theta.T, IM_mat)
			B_im = (self.theta.T).dot(self.theta)
			eit1 = time.perf_counter() - eit1
			
			if fixed_H_im is not None:
				self.phi_im = fixed_H_im
				self.phi_im_it = 0
			else:
				self.phi_im, self.phi_im_it = self._HALS(self.phi_im, B_im, A_im, eit1)

			error_expr = np.sqrt(nM_expr - 2*np.sum(self.phi_expr*A_expr) + np.sum(B_expr*(self.phi_expr.dot(self.phi_expr.T))))
			error_im =  np.sqrt(nM_im - 2*np.sum(self.phi_im*A_im) + np.sum(B_im*(self.phi_im.dot(self.phi_im.T))))
			          

			epoch_end =time.perf_counter()
			
			epoch_duration = epoch_end - epoch_start
			logging.info('epoch duration: {}\nloss: {}'.format(epoch_duration, error_expr+error_im))
			logging.info('theta expr iter: {}\nphi expr iter:{}\nphi image iter: {}\n'.format(theta_it, self.phi_expr_it, self.phi_im_it))
			self.epoch_times.append(epoch_duration)
			self.loss.append(error_expr+error_im)
			self.loss_expr.append(error_expr)
			self.loss_im.append(error_im)
			self.epoch_iter.append(theta_it + self.phi_expr_it + self.phi_im_it)
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

		# self.theta = theta
		self.phi_expr = pd.DataFrame(self.phi_expr)
		# self.phi_im = pd.DataFrame(phi_im)
		self.error_expr = error_expr
		self.error_im = error_im
		self.total_time = time.perf_counter() - start


		
		end = time.perf_counter()	
		# self.theta = theta
		


		# self.total_time = end - start
		# logging.info(self.total_time)
		l = []
		for i in range(0,len(self.gene_expr)):
			nor = NormalizeData(self.theta[i,:])
			l.append(nor)

		theta_nor = pd.DataFrame(l)

		print('Adding factor values to adata.obs ...')

		n = theta_nor.shape[1]
		
		for i in range(n):
			# adata.obs[f"Factor_{i+1}"] = theta_nor.iloc[:,i].values
			adata.obs[f"Factor_{i+1}"] = self.theta[:,i]


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
				if self.reg:
					deltaV = np.maximum((UtM[k, :] - UtU[k, :].dot(V) -self.l1_weight * np.ones(n)) /UtU[k, k], -V[k, :])
				else:
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
        		if self.reg:
        			deltaW = np.maximum((((mod1_skew*(rnaMHt[:, k] - W.dot(rnaHHt[:, k]))) +((2-mod1_skew)*(imgMHt[:, k] -W.dot(imgHHt[:, k]))) -(self.l1_weight*np.ones(n))) /
        				((mod1_skew*rnaHHt[k, k]) + ((2-mod1_skew)*imgHHt[k, k]))),-W[:, k])

	            #print(W.shape, imgHHt.shape)
	            #print(W.dot(imgHHt[:,k]).shape)
	        	else:
	        		deltaW = np.maximum(((mod1_skew*(rnaMHt[:,k] -W.dot(rnaHHt[:,k])) + (2-mod1_skew)*(imgMHt[:,k]-W.dot(imgHHt[:,k])))/((mod1_skew*rnaHHt[k, k]) + ((2-mod1_skew)*imgHHt[k,k])) ), -W[:, k])

	        	W[:, k] = W[:, k] + deltaW
	        	nodelta = nodelta + deltaW.dot(deltaW.T)
	        	W[W[:, k] == 0, k] = 1e-16*np.max(W) 
          

	            # deltaW = np.maximum(((mod1_skew*(rnaMHt[:,k] -W.dot(rnaHHt[:,k])) + (2-mod1_skew)*(imgMHt[:,k]-W.dot(imgHHt[:,k])))/
#             (lam*(np.dot(self.lap,W[:,k]))+(mod1_skew*rnaHHt[k, k]) + ((2-mod1_skew)*imgHHt[k,k]))), -W[:, k])


	        if cnt == 1:
	            eps0 = nodelta
	            eit3 = time.perf_counter() - eit3
	           
	        eps = nodelta
	        cnt = 0
	        n_it += 1
	        print(f"Iteration {n_it}, deltaW norm = {np.linalg.norm(deltaW)}, eps = {eps}")
	        
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
	def _initialize_nmf(self, X, IM_mat, n_components, fixed_W=None,fixed_H_expr=None,fixed_H_im=None, eps=1e-6,init='nnsdvd'):

			"""Algorithms for NMF initialization
			Computes an initial guess for the non-negative
			rank k matrix approximation for X: X = WH
			Parameters IM_mat_tfidf_log
			----------
			X : array-like, shape (n_samples, n_features)
				The data matrix to be decomposed.
			n_components : integer
				The number of components desired in the approximation.
			init :  None | 'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar' implemented as per sklearn NMF
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
			np.random.default_rng(self.random_state)

			n_samples, n_features = X.shape
	

			# if init == 'user_input':
			# 	return W_in, H_1_in, H_2_in

			if not sparse.issparse(X) and np.any(np.isnan(X)):
				raise ValueError("NMF initializations with NNDSVD are not available "
								 "with missing values (np.nan).")
								 
			if init == 'random':
				X_mean = X.mean()
				rng = check_random_state(random_state)
				avg = np.sqrt(X_mean /10 * n_components)
				avg_im = np.sqrt(IM_mat.mean() / n_components)
				
	
				if fixed_W is not None:
					W = fixed_W
				else:
					W = avg * rng.standard_normal(size=(n_samples, n_components)).astype(X.dtype, copy=False)
				if fixed_H_expr is not None:
					H = fixed_H_expr
				else:
					H = avg * rng.standard_normal(size=(n_components, X.shape[1])).astype(X.dtype, copy=False)

				if fixed_H_im is not None:
					H_im = fixed_H_im
				else:
					H_im = avg_im * rng.standard_normal(size=(n_components, IM_mat.shape[1])).astype(IM_mat.dtype, copy=False)


				# we do not write np.abs(H, out=H) to stay compatible with
				# numpy 1.5 and earlier where the 'out' keyword is not
				# supported as a kwarg on ufuncs
				np.abs(H, out=H)
				np.abs(W, out=W)
				np.abs(H_im, out=H_im)
				print(f"Random Initialization: X mean = {X_mean}")
				print(f"Random Initialization: H mean = {H.mean()}, H std = {H.std()}")
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
			
			if init == "nndsvd":

				    # Convert to dense if necessary
				# if sparse.issparse(X):
				# 	X = X.toarray()
				# if sparse.issparse(IM_mat):
				# 	IM_mat = IM_mat.toarray()
				#     # Remove mean of each feature
				# X_mean = X.mean(axis=0)
				# IM_mean = IM_mat.mean(axis=0)
				# X -= X_mean
				# IM_mat -= IM_mean
				# U, S, Vt = randomized_svd(X, n_components,random_state=random_state)
				U, S, Vt = svds(X, k=n_components, which='LM')
				U, Vt = U[:, ::-1], Vt[::-1, :]
				S = S[::-1]

		
				W = np.zeros(U.shape)
				H= np.zeros(Vt.shape)
				H_im = np.zeros((n_components, IM_mat.shape[1]))

			# The leading singular triplet is non-negative
			# so it can be used as is for initialization.
				W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])

				H[0, :] = np.sqrt(S[0]) * np.abs(Vt[0, :])

				for j in range(1, n_components):
					x, y = U[:, j], Vt[j, :]

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

				W[W < eps] = 0
				H[H < eps] = 0	

			
				H_im = safe_sparse_dot(np.linalg.pinv(W), IM_mat)
				H_im[H_im < eps] = 0

			 
			return W, H, H_im
	


def calculate_graph(adata):
	from scipy.sparse import csgraph

	G = sq.gr.spatial_neighbors(adata, n_rings=2,  coord_type= 'Grid', n_neighs=6)





# def extract_image_features(adata, u_img = False, img = None, layer = None, 
# 						library_id = None, features = 'histogram',
# 						mask_circle = False,
# 						scale_img = 1,
# 						spot_scale = 2.5, 
# 						key_added="features", show_progress_bar = True, n_jobs = None):

# 		"""Extract image features using squidpy.im.calculate_image_features function from Squidpy https://squidpy.readthedocs.io/en/stable/#
# 		:adata: anndata object with spatial data
# 		:layer: mage layer in img that should be processed. If None and only 1 layer is present, it will be selected.
# 		:library_id: if None, there should only exist one entry in anndata.AnnData.uns ['{spatial_key}'].
# 		If a str, first search anndata.AnnData.obs ['{library_id}'] which contains the mapping from observations to library ids, then search anndata.AnnData.uns ['{spatial_key}'].
# 		:features(str: Features to be calculated. Valid options are: 'texture', 'summary','histogram','segmentation','custom'
# 		:key_added(str):  Key in anndata.AnnData.obsm where to store the calculated features 
# 		:n_jobs(optional,int): Number of parallel jobs
# 		:show_progress_bar:– Whether to show the progress bar or not  
# 		"""
# 		from PIL import Image
  		
# 		library_id = list(adata.uns['spatial'].keys())[0]

# 		if  u_img == True:
# 			img = img
# 		else:
# 			# img: High-resolution image
# 			img = sq.im.ImageContainer(
# 				adata.uns["spatial"][library_id]["images"]["hires"],
# 				scale=adata.uns["spatial"][library_id]["scalefactors"]["tissue_hires_scalef"]
# 			)
# 		print('Extracting Image Features ...')
		
# 		features = sq.im.calculate_image_features(adata, img, features=features, mask_circle=mask_circle,scale = scale_img, spot_scale = spot_scale, key_added="features", show_progress_bar=True)
		
# 		adata.obsm[key_added].columns = ad.utils.make_index_unique(adata.obsm[key_added].columns)


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

	sel_clust = ['Factor_'+str(i+1) for i in range(topics)]
	with mpl.rc_context({'figure.figsize': (5, 6), 'axes.facecolor': 'black'}):
		sc.pl.spatial(adata,
				cmap='magma',
				color=sel_clust,
				ncols=5,
				size=1, img_key='hires', 
				alpha_img=0.1
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
	
