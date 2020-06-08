from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import func as func

if __name__ == '__main__':
    # load data
    data = pd.read_table('.\data\Synthetic_data\SC_2.txt', sep=' ', header=None)
    class_labels = pd.read_table('.\data\Synthetic_data\class_label.txt', sep=' ', header=None)

    true_labels = np.array(class_labels.iloc[:,0])

    # clustering : K-means & DBSCAN
    k_means_pred = KMeans(n_clusters=2, random_state=128).fit_predict(data)

    # determine the parameters (Eps, minPts) for DBSCAN
    # from sklearn.neighbors import NearestNeighbors
    # nbrs = NearestNeighbors(n_neighbors=4).fit(data)
    # distances, indices = nbrs.kneighbors(data)
    # distanceDec = sorted(distances[:, 3], reverse=False)
    # plt.plot(indices[:, 0], distanceDec)
    # plt.show()
    dbscan_pred = DBSCAN(eps=0.035, min_samples=4).fit_predict(data)

    # compute evaluation criteria: DBI,SC,MI,NMI,AMI
    print("Class Label")
    func.cal_int_ind(data, true_labels)

    print("K-means")
    func.cal_int_ind(data, k_means_pred)
    func.cal_ext_ind(true_labels, k_means_pred)

    print("DBSCAN")
    func.cal_int_ind(data, dbscan_pred)
    func.cal_ext_ind(true_labels, dbscan_pred)

    # plot clustering result
    y_pred = pd.DataFrame(k_means_pred)

    dt_0 = data[class_labels.iloc[:,0] == 0]
    y_pred_0 = y_pred[class_labels.iloc[:,0] == 0]
    dt_1 = data[class_labels.iloc[:,0] == 1]
    y_pred_1 = y_pred[class_labels.iloc[:,0] == 1]

    color_0 = func.cal_col(np.array(y_pred_0))
    color_1 = func.cal_col(np.array(y_pred_1))

    plt.figure()
    plt.tick_params(labelsize=13)

    plt.scatter(dt_0.iloc[:, 0], dt_0.iloc[:, 1], c='', s=50, marker='^', edgecolors=color_0)
    plt.scatter(dt_1.iloc[:, 0], dt_1.iloc[:, 1], c='', s=80, marker='.', edgecolors=color_1)
    # plt.legend(labels=['Class 1', 'Class 2'], loc=1, prop={'size': 15})
    # plt.plot([0.05, 0.9], [0.45, 0.7], color='k', linestyle='dashed', linewidth=2)

    plt.xticks([])  # remove the tick mark of x axis
    plt.yticks([])  # remove the tick mark of y axis

    #plt.savefig('results/Synthetic_data/Kmeans_2.png', dpi=1024)
    plt.show()

