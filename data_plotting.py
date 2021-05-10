import pickle
import numpy as np
import matplotlib.pyplot as plt
from main import distance


def plot_local_consumption(results, N):
    cases = ["recommendation", "no_recommendation", "oracle"]
    for case in cases:
        nb_items_chosen = len(results[case][0][0])
        consumption = []
        for t in range(1, nb_items_chosen):
            distances = []
            for population in results[case]:
                for user in population:
                    distances.append(distance(N, user[t-1], user[t]))
            consumption.append(np.mean(distances))
        plt.plot(range(1, nb_items_chosen), consumption, label=case)
    plt.xlabel("t")
    plt.ylabel("average distance")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    with open("results.pickle", "rb") as file:
        results = pickle.load(file)

    N = 200
    plot_local_consumption(results, 200)

