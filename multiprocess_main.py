import numpy as np
from numpy.random import multivariate_normal
import math
import pickle
import threading
import multiprocessing
#from queue import Queue


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
    cov11 = np.delete(np.delete(cov_matrix, chosen_items, 0), chosen_items, 1)
    cov22 = np.delete(np.delete(cov_matrix, left_over_items, 0), left_over_items, 1)
    cov21 = np.delete(np.delete(cov_matrix, chosen_items, 0), left_over_items, 1)
    cov12 = np.delete(np.delete(cov_matrix, left_over_items, 0), chosen_items, 1)
    return cov11, cov12, cov21, cov22


def update_utility(C_t, utility, mu_utility, cov_utility, N):
    chosen_items = C_t.copy()
    chosen_items.sort()
    left_over_items = [i for i in range(N) if i not in chosen_items]
    mu_chosen = np.array([mu_utility[i] for i in chosen_items])
    mu_left_over = np.array([mu_utility[i] for i in left_over_items])
    cov11, cov12, cov21, cov22 = covariance_split(chosen_items, left_over_items, cov_utility)

    c_utility = np.array([utility[i] for i in chosen_items])
    inv_cov22 = np.linalg.inv(cov22)
    temp = cov21 @ inv_cov22
    mu = mu_left_over + temp @ (c_utility - mu_chosen)
    cov = cov11 - temp @ cov12
    return mu, cov


def simulate(rho, beta, gamma, sigma, population_size, user_size, N, items_to_choose):
    results = dict()
    results["recommendation"] = []
    results["no_recommendation"] = []
    results["oracle"] = []

    sigma_ibar = 0.1
    rho_ibar = 0.0

    sigma_i = sigma
    cov_i = init_covariance_matrix(sigma_i, rho, N)
    cov = init_covariance_matrix(sigma, rho, N)
    cov_ibar = init_covariance_matrix(sigma_ibar, rho_ibar, N)

    # start simulation of 1 population,
    for pop_i in range(population_size):
        result_recommendation = []
        result_no_recommencation = []
        result_oracle = []
        V_bar = np.zeros(N)
        V = multivariate_normal(V_bar, cov)

        # start simulation of 1 user,
        for user_i in range(user_size):
            V_ibar = multivariate_normal(np.zeros(N), cov_ibar)
            mu_V = V_ibar
            V_i = multivariate_normal(V_ibar, cov_i)

            utility_i = V_i + (beta * V)
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
            result_oracle.append(oracle_C)
            # print(oracle_C)

            # no recommendation
            cov_utility = cov_i + math.pow(beta, 2) * cov  # no clue why this is needed (todo: find this out)

            C_no_recommendation = []
            for t in range(items_to_choose):
                if t == 0:
                    mu_utility_t = mu_utility_i
                    cov_utility_t = cov_utility
                else:
                    mu_utility_t, cov_utility_t = update_utility(C_no_recommendation, utility_i, mu_utility_i,
                                                                 cov_utility, N)

                certainty_equivalent_values = certainty_equivalent(gamma, mu_utility_t, cov_utility_t)
                left_over_items = [i for i in range(N) if i not in C_no_recommendation]
                item = left_over_items[np.argmax(certainty_equivalent_values)]
                C_no_recommendation.append(item)
            result_no_recommencation.append(C_no_recommendation)
            # print(C_no_recommendation)

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
            result_recommendation.append(C_recommendation)
            # print(C_recommendation)
        results["recommendation"].append(result_recommendation)
        results["no_recommendation"].append(result_no_recommencation)
        results["oracle"].append(result_oracle)
        print("population: ", pop_i)
    return results


class MyThread(threading.Thread):
    def __init__(self, rho, beta, gamma, sigma, population_size, user_size, N, items_to_choose, results):
        threading.Thread.__init__(self)
        self.rho = rho
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.population_size = population_size
        self.user_size = user_size
        self.N = N
        self.items_to_choose = items_to_choose
        self.results = results

    def run(self):
        result = simulate(self.rho, self.beta, self.gamma, self.sigma, self.population_size, self.user_size,
                 self.N, self.items_to_choose)
        threadLock.acquire()
        self.results[(self.rho, self.beta, self.gamma, self.sigma)] = result
        threadLock.release()


class QueThread(multiprocessing.Process):
    def __init__(self, jobs, population_size, user_size, N, items_to_choose, results):
        multiprocessing.Process.__init__(self)
        self.population_size = population_size
        self.user_size = user_size
        self.N = N
        self.items_to_choose = items_to_choose
        self.results = results
        self.jobs = jobs

    def run(self):
        while not self.jobs.empty():
            rho, beta, gamma, sigma = self.jobs.get()
            result = simulate(rho, beta, gamma, sigma, self.population_size, self.user_size, self.N, self.items_to_choose)
            threadLock.acquire()
            self.results[(rho, beta, gamma, sigma)] = result
            threadLock.release()
            self.jobs.task_done()


threadLock = multiprocessing.Lock()

if __name__ == "__main__":
    results = dict()
    N = 200
    pop_size = 100
    user_size = 100
    items_to_choose = 20  # same as T in the original code
    rhos = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
    betas = [0, 0.4, 0.8, 1, 2, 5]
    gammas = [0, 0.3, 0.6, 1, 5]  # this is the same as alpha in the original code
    sigmas = [0.25, 0.5, 1, 2, 4]
    threads = []
    jobs = multiprocessing.JoinableQueue()
    for rho in rhos:
        for beta in betas:
            for gamma in gammas:
                for sigma in sigmas:
                    jobs.put((rho, beta, gamma, sigma))
                    # thread = MyThread(rho, beta, gamma, sigma, pop_size, user_size, N, items_to_choose, results)
                    # thread.start()
                    # threads.append(thread)
    for i in range(16):
        worker = QueThread(jobs, pop_size, user_size, N, items_to_choose, results)
        worker.start()

    # for t in threads:
    #     t.join()
    jobs.join()

    with open("results.pickle", "wb") as file:
        pickle.dump(results, file)




