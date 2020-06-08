from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sklearn.preprocessing as pro
from sklearn.preprocessing import LabelEncoder

import func as func

if __name__ == '__main__':
    # load data
    df = pd.read_excel('.\data\Height_weight\weight-height.xlsx')

    df1 = df[0:5000].sample(200, replace=False, random_state=0)
    df2 = df[5000:].sample(200, replace=False, random_state=1)
    df = df1.append(df2)
    df = df.reset_index(drop=True)

    data = df.loc[:,['Height','Weight']]
    class_labels = df.loc[:,['Gender']]
    true_labels = LabelEncoder().fit_transform(class_labels)

    min_max_scaler = pro.MinMaxScaler() # normalization to [0,1]
    data = min_max_scaler.fit_transform(data)

    # clustering : K-means & DBSCAN
    k_means_pred = KMeans(n_clusters=2, random_state=128).fit_predict(data)

    # determine the parameters (Eps, minPts) for DBSCAN
    # from sklearn.neighbors import NearestNeighbors
    # nbrs = NearestNeighbors(n_neighbors=4).fit(data)
    # distances, indices = nbrs.kneighbors(data)
    # distanceDec = sorted(distances[:, 3], reverse=False)
    # plt.plot(indices[:, 0], distanceDec)
    # plt.show()
    dbscan_pred = DBSCAN(eps=0.070, min_samples=4).fit_predict(data)

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

    dt_0 = data[class_labels['Gender']=='Male']
    y_pred_0 = y_pred[class_labels['Gender']=='Male']
    dt_1 = data[class_labels['Gender']=='Female']
    y_pred_1 = y_pred[class_labels['Gender']=='Female']

    plt.figure()
    plt.tick_params(labelsize=13)

    color_0 = func.cal_col(np.array(y_pred_0))
    color_1 = func.cal_col(np.array(y_pred_1))
    plt.scatter(dt_0[:,0], dt_0[:,1], c='', s=50, marker='^', edgecolors=color_0)
    plt.scatter(dt_1[:,0], dt_1[:,1], c='', s=80, marker='.', edgecolors=color_1)
    #plt.legend(labels=['Male', 'Female'], loc=2, prop={'size': 15})

    plt.xticks([])  # remove the tick mark of x axis
    plt.yticks([])  # remove the tick mark of y axis
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20,
             }
    plt.xlabel('Height', fontdict=font2)
    plt.ylabel('Weight', fontdict=font2)
    #plt.savefig('results/Height_weight/Kmeans_2.png', dpi=1024)
    plt.show()
