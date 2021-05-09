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


def covariance_split(chosen_items, left_over_items, cov_matrix):
    nb_chosen = len(chosen_items)
    nb_left_over = len(left_over_items)
    cov11 = np.zeros((nb_left_over, nb_left_over))
    cov22 = np.zeros((nb_chosen, nb_chosen))
    cov12 = np.zeros((nb_chosen, nb_left_over))
    cov21 = np.zeros((nb_left_over, nb_chosen))

    for i in range(nb_chosen):
        for j in range(nb_chosen):
            cov22[i][j] = cov_matrix[chosen_items[i]][chosen_items[j]]
        for j in range(nb_left_over):
            cov21[j][i] = cov_matrix[chosen_items[i]][left_over_items[j]]
            cov12[i][j] = cov_matrix[left_over_items[j]][chosen_items[i]]
    for i in range(nb_left_over):
        for j in range(nb_left_over):
            cov11[i][j] = cov_matrix[left_over_items[i]][left_over_items[j]]

    return cov11, cov12, cov21, cov22


def update_utility(C_t, utility, mu_utility, cov_utility, N):
    chosen_items = C_t
    left_over_items = [i for i in range(N) if i not in C_t]
    mu_chosen = np.array([mu_utility[i] for i in chosen_items])
    mu_left_over = np.array([mu_utility[i] for i in left_over_items])
    cov11, cov12, cov21, cov22 = covariance_split(chosen_items, left_over_items, cov_utility)

    c_utility = np.array([utility[i] for i in chosen_items])
    inv_cov22 = np.linalg.inv(cov22)
    temp = cov21@inv_cov22
    mu = mu_left_over + temp@(c_utility-mu_chosen)
    cov = cov11 - temp@cov12
    return mu, cov


if __name__ == "__main__":
    N = 200
    items_to_choose = 20  # same as T in the original code
    rho = 0.1
    beta = 0.4
    gamma = 0.3  # this is the same as alpha in the original code
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
    mu_utility_i = mu_V + beta * V_bar

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
    oracle_C = list(oracle_C)
    print(oracle_C)

    # no recommendation
    cov_utility = cov_i + math.pow(beta, 2) * cov  # no clue why this is needed (todo: find this out)

    C_no_recommendation = []
    for t in range(items_to_choose):
        if t == 0:
            mu_utility_t = mu_utility_i
            cov_utility_t = cov_utility
        else:
            mu_utility_t, cov_utility_t = update_utility(C_no_recommendation, utility_i, mu_utility_i, cov_utility, N)

        certainty_equivalent_values = certainty_equivalent(gamma, mu_utility_t, cov_utility_t)
        left_over_items = [i for i in range(N) if i not in C_no_recommendation]
        item = left_over_items[np.argmax(certainty_equivalent_values)]
        C_no_recommendation.append(item)

    print(C_no_recommendation)

    # recommendation
    C_recommendation = []
    current_V = V
    for t in range(items_to_choose):
        if t == 0:
            mu_V_t = V_ibar
            cov_i_t = cov_i
        else:
            mu_V_t, cov_i_t = update_utility(C_recommendation, V_i, V_ibar, cov_i, N)
            current_V = np.array([V[i] for i in range(N) if i not in C_recommendation])
        mu_utility_t = mu_V_t + beta * current_V

        certainty_equivalent_values = certainty_equivalent(gamma, mu_utility_t, cov_i_t)
        left_over_items = [i for i in range(N) if i not in C_recommendation]
        item = left_over_items[np.argmax(certainty_equivalent_values)]
        C_recommendation.append(item)

    print(C_recommendation)




