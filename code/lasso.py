"""
Functions for 1st Assignment in FMAN45 regarding Lasso
"""

import numpy as np
from scipy.signal.windows import hann
from numpy.random import permutation
from sklearn.utils import shuffle
np.random.seed(42) 

def lasso_ccd(data, reg_matrix, lambda_val, w_old=None):
    np.random.seed(42) 
    N, M = reg_matrix.shape
    if data.shape[0] != N:
        raise ValueError("Mismatch in number of samples between data and regression matrix.")
    
    if w_old is None:
        w_old = np.zeros((M, 1))
    
    w_hat = w_old.copy()
    res = data - reg_matrix @ w_hat
    w_ind = (np.abs(w_hat) > lambda_val).astype(np.int32)

    nbr_iter = 50
    update_cycle = 5

    for i in range(nbr_iter):
        if (i % update_cycle) and i > 1:
            ind_rand_order = permutation(np.where(w_ind)[0])
        else:
            ind_rand_order = permutation(M)

        for idx in ind_rand_order:
            x_i = reg_matrix[:, idx:idx+1]  # (N,1)
            res += x_i * w_hat[idx]         # put old back in

            x_tr = (x_i.T @ res).item()

            #x_tr = float(x_i.T @ res)
            a = float(x_i.T @ x_i)

            if abs(x_tr) > lambda_val:
                w_hat[idx] = (abs(x_tr) - lambda_val) * np.sign(x_tr) / a
            else:
                w_hat[idx] = 0.0

            res -= x_i * w_hat[idx]        # remove new from res
            w_ind[idx] = int(abs(w_hat[idx]) > lambda_val)
    
    return w_hat

def lasso_cv(t, X, lambda_vec, nbr_folds):
    """K-fold cross-validation for LASSO"""
    np.random.seed(42) 
    N = X.shape[0]
    indices = np.arange(N)
    indices = shuffle(indices, random_state=42)
    fold_size = N // nbr_folds

    se_val = np.zeros((nbr_folds, len(lambda_vec)))
    se_est = np.zeros((nbr_folds, len(lambda_vec)))

    for k in range(nbr_folds):
        val_idx = indices[k*fold_size:(k+1)*fold_size]
        est_idx = np.setdiff1d(indices, val_idx)

        X_val = X[val_idx]
        t_val = t[val_idx]
        X_est = X[est_idx]
        t_est = t[est_idx]

        for j, lambda_val in enumerate(lambda_vec):
            w_hat = lasso_ccd(t_est, X_est, lambda_val)
            y_val = X_val @ w_hat
            y_est = X_est @ w_hat
            se_val[k, j] = np.mean((t_val - y_val)**2)
            se_est[k, j] = np.mean((t_est - y_est)**2)

    rmse_val = np.sqrt(np.mean(se_val, axis=0))
    rmse_est = np.sqrt(np.mean(se_est, axis=0))
    lambda_opt = lambda_vec[np.argmin(rmse_val)]
    w_opt = lasso_ccd(t, X, lambda_opt)

    return w_opt, lambda_opt, rmse_val, rmse_est


def multiframe_lasso_cv(data, reg_matrix, lambda_vec, nbr_folds):
    """
    Calculates the LASSO solution for all frames and trains the hyperparameter using
    cross-validation.

    Parameters
    data        - Nx1 data vector
    reg_matrix  - NxM regression matrix (ndarray)
    lambda_vec  - vector grid of possible hyperparameters
    nbr_folds   - number of folds

    Return:
    W_opt       - MxN frames LASSO estimate for optimal lambda
    lambda_opt  - Optimal lambda value
    rmse_val    - Vector of validation MSE values for lambdas in grid
    rmse_est    - Vector of estimation MSE values for lambdas in grid
    """
    n_frames = train_frames.shape[0]
    M = Xaudio.shape[1]
    
    W_opt = np.zeros((M, n_frames))
    lambda_opt = np.zeros(n_frames)
    rmse_val = np.zeros((n_frames, len(lambda_vec)))
    rmse_est = np.zeros((n_frames, len(lambda_vec)))
    
    for i in range(n_frames):
        t = train_frames[i]
        w_hat, l_opt, val_rmse, est_rmse = lasso_cv(t, Xaudio, lambda_vec, n_folds)
        W_opt[:, i] = w_hat
        lambda_opt[i] = l_opt
        rmse_val[i, :] = val_rmse
        rmse_est[i, :] = est_rmse
    
    return W_opt, lambda_opt, rmse_val, rmse_est





def lasso_denoise(Tnoisy, X, lambda_val):
    """
    Denoises the data in Tnoisy using LASSO estimates for hyperparameter lambdaopt.
    Cycles through the frames in Tnoisy, calculates the LASSO estimate, selecting
    the non-zero components and reconstructing the data using these components only,
    using a WOLS estimate, weighted by the Hanning window.

    Parameters:
    - Tnoisy (numpy.ndarray): NNx1 noisy data vector
    - X (numpy.ndarray): NxM regression matrix
    - lambda_val (float): Hyperparameter value (selected from cross-validation)

    Returns:
    - Yclean (numpy.ndarray): NNx1 denoised data vector
    """

    # Sizes
    NN = len(Tnoisy)
    N, M = X.shape

    # Frame indices parameters
    loc = 0
    hop = N // 2
    idx = np.arange(N)

    Z = np.diag(hann(N))  # Weight matrix
    Yclean = np.zeros_like(Tnoisy)  # Clean data preallocation

    while loc + N <= NN:
        t = Tnoisy[loc + idx]  # Pick out data in the current frame
        wlasso = lasso_ccd(t, X, lambda_val)  # Calculate LASSO estimate
        nzidx = np.abs(wlasso.reshape(-1)) > 0  # Find nonzero indices

        # Calculate weighted OLS estimate for nonzero indices
        wols = np.linalg.lstsq(Z @ X[:, nzidx], Z @ t, rcond=None)[0]
        # Reconstruct denoised signal
        Yclean[loc + idx] += Z @ X[:, nzidx] @ wols

        loc += hop  # Move indices for the next frame
        print(f"{int(loc / NN * 100)} %")  # Show progress

    print("100 %")
    return Yclean
