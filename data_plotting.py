import pickle
import numpy as np
import matplotlib.pyplot as plt
from main import distance


def plot_local_consumption(results, N):
    cases = ["recommendation", "no_recommendation", "oracle"]
    nb_items_chosen = 20
    for case in cases:
        consumption = []
        for t in range(1, nb_items_chosen):
            distances = []
            for params in results:
                result = results[params]
                rho, beta, gamma, sigma = params
                for population in result[case]:
                    for user in population:
                        distances.append(distance(N, user[t-1], user[t]))
            consumption.append(np.mean(distances))
            print("case: ", case, "\t t:", t)
        plt.plot(range(1, nb_items_chosen), consumption, label=case)

    plt.ylim(bottom=0)
    plt.xlabel("t")
    plt.ylabel("average distance")
    plt.legend()
    plt.show()


def plot_local_consumption_gamma(results, N):
    nb_items_chosen = 20
    for gamma in gammas:
        consumption = []
        for t in range(1, nb_items_chosen):
            distances = []
            for rho in rhos:
                for beta in betas:
                    for sigma in sigmas:
                        result = results[(rho, beta, gamma, sigma)]
                        for population in result["recommendation"]:
                            for user in population:
                                distances.append(distance(N, user[t-1], user[t]))
            consumption.append(np.mean(distances))
            print(t)
        plt.plot(range(1, nb_items_chosen), consumption, label=f'gamma={gamma}')
    plt.ylim(bottom=0)
    plt.xlabel("t")
    plt.ylabel("average distance")
    plt.legend()
    plt.show()


def diversity(items):
    distances = []
    N = 200
    for i, item in enumerate(items):
        for j in range(i, len(items)):
            distances.append(distance(N, item, items[j]))
    result = (1/(N*20*19*0.5))*sum(distances)
    return result


def plot_item_diversity(results, N):
    cases = ["recommendation", "no_recommendation", "oracle"]
    temp = dict()

    for case in cases:
        print("starting case: ", case)
        for i in rhos:
            temp[i] = []
        for params in results:
            rho, beta, gamma, sigma = params
            result = results[params]
            print(params)
            for population in result[case]:
                for user in population:
                    temp[rho].append(diversity(user))
        y_values = []
        for i in rhos:
            y_values.append(np.mean(temp[i]))
        plt.plot(rhos, y_values, label=case)

    plt.ylim(bottom=0)
    plt.xlabel("rho")
    plt.ylabel("item diversity")
    plt.legend()
    plt.show()

def plot_item_welfare(results, N):
    cases = ["recommendation_W", "no_recommendation_W", "oracle_W"]
    temp = dict()

    for case in cases:
        print("starting case: ", case)
        for i in rhos:
            temp[i] = []
        for params in results:
            rho, beta, gamma, sigma = params
            result = results[params]
            print(params)
            for population in result[case]:
                for user in population:
                    temp[rho].append(np.mean(user))
        y_values = []
        for i in rhos:
            y_values.append(np.mean(temp[i]))
        plt.plot(rhos, y_values, label=case)
    plt.ylim(bottom=0)
    plt.xlabel("rho")
    plt.ylabel("item diversity")
    plt.legend()
    plt.show()

def jaccard_index(s1, s2):
    size_s1 = len(s1)
    size_s2 = len(s2)
    intersect = set(s1) & set(s2)
    size_in = len(intersect)
    jaccard_in = (size_in/ (size_s1 + size_s2 - size_in))
    return jaccard_in

def jaccard_distance(s1, s2):
    jaccard_dist = 1 - jaccard_index(s1, s2)
    return jaccard_dist

def homogeneity(users):
    jaccard_distances = []
    for i, user in enumerate(users):
        for j in range(i, len(users)):
            jaccard_distances.append(jaccard_index(user, users[j]))
    homogeneity = ((1/4950)*sum(jaccard_distances))
    return homogeneity

def plot_homogeneity(results):
    cases = ["recommendation", "no_recommendation", "oracle"]
    temp = dict()

    for case in cases:
        print("starting case: ", case)
        for i in rhos:
            temp[i] = []
        for params in results:
            rho, beta, gamma, sigma = params
            result = results[params]
            print(params)
            for population in result[case]:
                temp[rho].append(homogeneity(population))
        y_values = []
        for i in rhos:
            y_values.append(np.mean(temp[i]))
        plt.plot(rhos, y_values, label=case)

    plt.ylim(bottom=0)
    plt.xlabel("rho")
    plt.ylabel("homogeneity")
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    rhos = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
    betas = [0, 0.4, 0.8, 1, 2, 5]
    gammas = [0, 0.3, 0.6, 1, 5]  # this is the same as alpha in the original code
    sigmas = [0.25, 0.5, 1, 2, 4]

    with open("results.pickle", "rb") as file:
        results = pickle.load(file)
    # with open("results_welfare.pickle", "rb") as file:
    #     results = pickle.load(file)
    # results are structured as follows: dictionary that maps 4 tuples representing the parameters to each result.
    # Each result is a dictionary with 3 entries ("recommendation", "no_recommendation", "oracle").
    # Each entry is a 100*100*20 matrix containing the chosen items in order of every single user in every population.
    # the first dimension represents the population number, second dimension represents the user number and the last dimension represents the chosen item.
    # so results[(0, 0, 0, 0)]["recommendation"][1][2][3] represents the 4th item chosen by the third user in the second population for the case of recommendation with parameters (0, 0, 0, 0).


    N = 200
    #plot_local_consumption(results, N)
    #plot_item_diversity(results, N)
    #plot_local_consumption_gamma(results, N)
    #plot_item_welfare(results, N)
    plot_homogeneity(results)
