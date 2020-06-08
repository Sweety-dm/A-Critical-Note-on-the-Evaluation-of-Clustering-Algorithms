import numpy as np
from scipy.spatial.distance import pdist
import math

import matplotlib.pyplot as plt


if __name__ == '__main__':

    n_trials = 10000
    r = 0.01
    clusters_num = np.arange(2,100,1)
    p_noverlap_eval = []
    p_noverlap_theo = []

    for num in clusters_num:
        overlap_num = 0
        for i_episode in range(n_trials):
            cluster_centers = np.random.random(size=[num, 2])
            dist = pdist(cluster_centers, metric='euclidean')
            if True in (dist < 2*r) :
                overlap_num += 1

        if (num%2==0) :
            p_overlap_eval = overlap_num/n_trials
            p_noverlap_eval.append(1-p_overlap_eval)

        p_overlap_theo = math.e ** (-2*num*(num-1)*math.pi*r*r)
        p_noverlap_theo.append(p_overlap_theo)

    # plot
    plt.figure(figsize=(7,5.5))
    plt.tick_params(labelsize=13)

    plt.plot(clusters_num, p_noverlap_theo, c='red', linewidth=3)
    plt.scatter(np.arange(2,100,2), p_noverlap_eval, marker='^', facecolors='none', edgecolors='green', s=60)

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20,
             }
    plt.ylabel('Probability (p)', fontdict=font2)
    plt.xlabel('Number of classes (k)', fontdict=font2)
    plt.legend(labels=['theoretical', 'experimental'], loc=1, prop={'size': 15})

    # plt.savefig('results/Discussion/overlap_prob.png', dpi=1024)
    plt.show()

