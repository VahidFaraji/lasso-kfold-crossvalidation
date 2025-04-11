"""
Module to run all tasks regarding first assignment.

How you manage this module is not as important. You can choose whether you do as is written in the
skeleton or if you prefer to have input-values from the command line. However, keep the names of
all methods in the lasso-module identical as we can then correct the assignment easier.
"""
import numpy as np
import argparse
from scipy.io import loadmat
from lasso import lasso_ccd 
from lasso import lasso_cv
import matplotlib.pyplot as plt


def task4():
    """
    Runs code for task 4
    """
    # Load data from .mat file
    data = loadmat("A1_data.mat")
    t = data["t"]        # Nx1 target vector
    X = data["X"]        # NxM regression matrix

    # Choose a regularization parameter
    lambda_val = 0.1

    # Run LASSO coordinate descent
    w_hat = lasso_ccd(t, X, lambda_val)

    # Print or check result (optional)
    print("Task 4")
    print("LASSO result (first 10 weights):")
    print(w_hat[:10])



def task5():

    data = loadmat("A1_data.mat")
    t = data["t"]
    X = data["X"]

    lambda_vec = np.logspace(-2, 2, 30)  # 0.01 to 100
    nbr_folds = 5

    w_opt, lambda_opt, rmse_val, rmse_est = lasso_cv(t, X, lambda_vec, nbr_folds)

    print(f"Task 5 - Optimal λ: {lambda_opt:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(lambda_vec, rmse_val, label="Validation RMSE", marker='o')
    plt.plot(lambda_vec, rmse_est, label="Estimation RMSE", marker='s')
    plt.axvline(x=lambda_opt, color='r', linestyle='--', label=f"Optimal λ = {lambda_opt:.4f}")
    plt.xscale("log")
    plt.xlabel("λ (log scale)")
    plt.ylabel("RMSE")
    plt.title("K-Fold Cross-Validation for LASSO")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def task6():
    from lasso import multiframe_lasso_cv
    from scipy.io import loadmat
    import matplotlib.pyplot as plt
    import numpy as np

    # Load data
    data = loadmat("A1_data.mat")
    Ttrain = data["Ttrain"].flatten()
    Xaudio = data["Xaudio"]

    # Define lambda grid and folds
    lambda_vec = np.logspace(-2, 2, 30)
    nbr_folds = 5

    # Run multiframe CV
    W_opt, lambda_opt, rmse_val, rmse_est = multiframe_lasso_cv(Ttrain, Xaudio, lambda_vec, nbr_folds)

    # Plot RMSE vs lambda
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_vec, rmse_val, label="Validation RMSE", marker='o')
    plt.plot(lambda_vec, rmse_est, label="Estimation RMSE", marker='s')
    plt.axvline(x=lambda_opt, color='r', linestyle='--', label=f"Optimal λ = {lambda_opt:.4f}")
    plt.xscale("log")
    plt.xlabel("λ (log scale)")
    plt.ylabel("RMSE")
    plt.title("Multiframe LASSO Cross-Validation (Task 6)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("task6_rmse_plot.png", dpi=300)
    plt.show()

    print(f"Task 6 - Optimal lambda: {lambda_opt:.4f}")


def task7():
    """
    Runs code for task 7
    """

    print("Task 7")


def main():
    """
    Runs a specified task given input from the user
    """

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-t",
        "--task",
        choices=["4", "5", "6", "7"],
        help="Runs code for selected task.",
    )
    args = parser.parse_args()
    try:
        if args.task is None:
            task = 0
        else:
            task = int(args.task)
    except ValueError:
        print("Select a valid task number")
        return

    if task == 4:
        task4()
    elif task == 5:
        task5()
    elif task == 6:
        task6()
    elif task == 7:
        task7()
    else:
        raise ValueError("Select a valid task number")


if __name__ == "__main__":
    main()
