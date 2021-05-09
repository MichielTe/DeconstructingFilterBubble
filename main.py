import numpy as np
from numpy.random import multivariate_normal
import math


def distance(N, i, j):
    return min(abs(i-j), N-abs(i-j))


def init_covariance_matrix(sigma, rho, N):
    cov = np.zeros(shape=(N, N))
    for i in range(N):
        for j in range(N):
            cov[i][j] = sigma*math.pow(rho, distance(N, i, j))
    return cov


def certainty_equivalent(gamma, mu, cov_matrix):
    return mu - 0.5*gamma*np.diagonal(cov_matrix)


if __name__ == "__main__":
    N = 200
    items_to_choose = 20  # same as T in the original code
    rho = 0.1
    beta = 0.4
    alpha = 0.3
    sigma = 0.25
    total_individuals = 100
    sigma_ibar = 0.1
    rho_ibar = 0.0

    sigma_i = sigma
    cov_i = init_covariance_matrix(sigma_i, rho, N)
    cov = init_covariance_matrix(sigma, rho, N)
    cov_ibar = init_covariance_matrix(sigma_ibar, rho_ibar, N)

    # start simulation of 1 population, TODO: loop this for all populations
    V_bar = np.zeros(N)
    V = multivariate_normal(V_bar, cov)

    # start simulation of 1 user, TODO: loop this for all users
    V_ibar = multivariate_normal(np.zeros(N), cov_ibar)
    mu_V = V_ibar
    V_i = multivariate_normal(V_ibar, cov_i)

    utility_i = V_i + (beta*V)
    mu_utility_i = V_ibar + math.pow(beta, 2) * cov
    print(utility_i)

    # oracle recommendation
    # oracle_C = []  # same implementation as original code
    # for t in range(items_to_choose):
    #     left_over_items = [i for i in range(N) if i not in oracle_C]
    #     Utility_of_items = [utility_i[n] for n in left_over_items]
    #     item = left_over_items[np.argmax(Utility_of_items)]
    #     oracle_C.append(item)

    item_utility_pairs = list(zip(range(N), utility_i))  # more efficient implementation
    item_utility_pairs.sort(key=lambda x: x[1], reverse=True)
    temp = item_utility_pairs[:items_to_choose]
    oracle_C, _ = zip(*temp)
    print(oracle_C)
