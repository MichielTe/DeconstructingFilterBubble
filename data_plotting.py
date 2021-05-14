import pickle
import numpy as np
import matplotlib.pyplot as plt
from main import distance


def plot_local_consumption(results, N):
    cases = ["recommendation", "no_recommendation", "oracle"]
    nb_items_chosen = 20
    for case in cases:
        consumption = []
        distances = []
        for t in range(1, nb_items_chosen):
            for params in results:
                result = results[params]
                for population in result[case]:
                    for user in population:
                        distances.append(distance(N, user[t-1], user[t]))
            consumption.append(np.mean(distances))
            print("case: ", case, "\t t:", t)
        plt.plot(range(1, nb_items_chosen), consumption, label=case)
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
    result = (1/(N*20*19))*sum(distances)
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

    plt.xlabel("rho")
    plt.ylabel("item diversity")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    rhos = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
    betas = [0, 0.4, 0.8, 1, 2, 5]
    gammas = [0, 0.3, 0.6, 1, 5]  # this is the same as alpha in the original code
    sigmas = [0.25, 0.5, 1, 2, 4]

    with open("results.pickle", "rb") as file:
        results = pickle.load(file)

    N = 200
    #plot_local_consumption(results, N)
    plot_item_diversity(results, N)

