import numpy as np
import pandas as pd
import os
import scanpy as sc
from matplotlib import pyplot as plt
import anndata as ad
from cellpie_main import intNMF
import pickle as pkl
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler



def preprocess_data_visium(adata, min_cells=100):

    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    scaler = MinMaxScaler()
    scaled=scaler.fit(adata.obsm['features'])
    adata.obsm['features'] = scaled.transform(adata.obsm['features'])

    return adata

def preprocess_data_visiumHD(adata):

    scaler = MinMaxScaler()
    scaled=scaler.fit(adata.obsm['features'])
    adata.obsm['features'] = scaled.transform(adata.obsm['features'])

    return adata


def center_data(adata):

    if type(adata.X).__name__ in ['csr_matrix', 'SparseCSRView']:
        gene_expr = pd.DataFrame(data = adata.X.toarray(), index = adata.obs.index, columns=adata.var_names)
    else:
        gene_expr = pd.DataFrame(data = adata.X, index = adata.obs.index, columns=adata.var_names)

    adata.layers["counts"] = adata.X.copy()
    
    gene_means =gene_expr.mean(axis=0)
    gene_expr_cent -= gene_means

    img_means =  image_features.mean(axis=0)
    image_features -= img_means

    return adata


def model_selection(adata, n_topics: list, mod1_skew=1,epochs=20,init= 'random',random_state=123):
    
    all_errors = pd.DataFrame()

    np.random.default_rng(random_state)
    errors = []
    mean = []
    n_samples, n_features = adata.shape

    r = n_samples // 2
    s = n_features // 2

    A_indices_rows = np.arange(r)
    B_indices_rows = np.arange(r)
    C_indices_rows = np.arange(r, n_samples)
    D_indices_rows = np.arange(r, n_samples)

    A_indices_cols = np.arange(s)
    B_indices_cols = np.arange(s, n_features)
    C_indices_cols = np.arange(s)
    D_indices_cols = np.arange(s, n_features)

    A = adata[A_indices_rows, :][:, A_indices_cols]
    B = adata[B_indices_rows, :][:, B_indices_cols]
    C = adata[C_indices_rows, :][:, C_indices_cols]
    D = adata[D_indices_rows, :][:, D_indices_cols]
    
    for k in n_topics:
            # Fit the model on block D
        np.random.default_rng(random_state)
        print(mod1_skew)
        model_D = intNMF(D, n_topics=k, random_state=random_state, init=init,epochs=epochs,mod1_skew=mod1_skew)
        model_D.fit(D)

        W_D = model_D.theta
        H_D = model_D.phi_expr.values

        model_B = intNMF(B, n_topics=k, random_state=random_state, init=init,epochs=epochs,mod1_skew=mod1_skew)
        model_B.fit(B, fixed_H_expr=H_D)  # Fix H to H_D
        W_B = model_B.theta

        model_C = intNMF(C, n_topics=k, random_state=random_state, init=init,epochs=epochs,mod1_skew=mod1_skew)
        model_C.fit(C, fixed_W=W_D)  # Fix W to W_D
        H_C = model_C.phi_expr.values

        A_pred = W_B @ H_C

        A_data = A.X.toarray()
        if np.isnan(A_data).any():
            print("Warning: A contains NaN values. Replacing NaNs with 0.")


        if np.isnan(A_pred).any():
            print("Warning: A_pred contains NaN values. Replacing NaNs with 0.")

        error = mean_squared_error(A.X.toarray(), A_pred)
        errors.append(error)
        error_mean = np.mean(errors)
        mean.append((k, error_mean))
        
    error_df = pd.DataFrame(mean, columns=['k', 'mean_error'])
    error_df.index = error_df['k']
    all_errors['factor'] = error_df['mean_error']
    
    for col in all_errors.columns:
        plt.plot(all_errors.index, all_errors[col], label=col)
    
    plt.xlabel('k')
    plt.ylabel('mean_error')
    plt.legend()
    plt.show()

    print('The optimal number of factors for each rep is:', all_errors.idxmin())

    return all_errors.idxmin(), all_errors



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
