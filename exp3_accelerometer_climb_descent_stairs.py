from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pandas as pd

import sklearn.preprocessing as pro
from sklearn.preprocessing import LabelEncoder

import func as func

if __name__ == '__main__':
    # load data
    df1 = pd.read_table('.\data\Accelerometer_climb_descent_stairs\climb_stairs.txt', sep=' ', header=None)
    df2 = pd.read_table('.\data\Accelerometer_climb_descent_stairs\descend_stairs.txt', sep=' ', header=None)
    df1.columns = ['x','y','z']
    df2.columns = ['x','y','z']
    df1['ADL'] = 'climb_stairs'
    df2['ADL'] = 'descend_stairs'
    df = df1.append(df2)
    df = df.reset_index(drop=True)

    data = df.loc[:, ['x', 'y', 'z']]
    class_labels = df.loc[:, ['ADL']]
    true_labels = LabelEncoder().fit_transform(class_labels)

    min_max_scaler = pro.MinMaxScaler()  # normalization to [0,1]
    data = min_max_scaler.fit_transform(data)

    # clustering : K-means & DBSCAN
    k_means_pred = KMeans(n_clusters=2, random_state=128).fit_predict(data)

    # determine the parameters (Eps, minPts) for DBSCAN
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=5).fit(data)
    distances, _ = nbrs.kneighbors(data)
    distanceDec = sorted(distances[:, 4], reverse=False)
    indices = range(len(distanceDec))
    plt.plot(indices, distanceDec)
    plt.show()
    dbscan_pred = DBSCAN(eps=0.11, min_samples=5).fit_predict(data)

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
    y_pred = pd.DataFrame(dbscan_pred)

    dt_0 = data[class_labels['ADL'] == 'climb_stairs']
    y_pred_0 = y_pred[class_labels['ADL'] == 'climb_stairs']
    dt_1 = data[class_labels['ADL'] == 'descend_stairs']
    y_pred_1 = y_pred[class_labels['ADL'] == 'descend_stairs']

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_ylim(0, 0.9)
    ax.set_zlim(0, 0.9)
    ax.view_init(elev=40.,azim=135)

    color_0 = func.cal_col(np.array(y_pred_0))
    color_1 = func.cal_col(np.array(y_pred_1))
    ax.scatter(dt_0[:, 0], dt_0[:, 1], dt_0[:, 2], c=color_0, s=50, marker='^')
    ax.scatter(dt_1[:, 0], dt_1[:, 1], dt_1[:, 2], c=color_1, s=80, marker='.')
    #plt.legend(labels=['climb_stairs', 'descend_stairs'], loc=1, prop={'size': 15})

    ax.set_xticklabels([])  # remove the tick mark of x axis
    ax.set_yticklabels([])  # remove the tick mark of y axis
    ax.set_zticklabels([])  # remove the tick mark of z axis

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20,
             }
    ax.set_xlabel('x', fontdict=font2)
    ax.set_ylabel('y', fontdict=font2)
    ax.set_zlabel('z', fontdict=font2)
    #plt.savefig('results/Accelerometer_climb_descent_stairs/DBSCAN_0.010_5.png', dpi=1024)
    plt.show()

