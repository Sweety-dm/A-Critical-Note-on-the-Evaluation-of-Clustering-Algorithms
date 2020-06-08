from sklearn import metrics

# internal criteria
def cal_int_ind(dt, y_pred):
    print("DBI: ", metrics.davies_bouldin_score(dt, y_pred))
    print("SC: ", metrics.silhouette_score(dt, y_pred))

# external criteria
def cal_ext_ind(flag, y_pred):
    print('ARI：', metrics.adjusted_rand_score(flag, y_pred))
    print('MI：', metrics.mutual_info_score(flag, y_pred))
    print('NMI：', metrics.normalized_mutual_info_score(flag, y_pred, average_method='arithmetic'))
    print('AMI：', metrics.adjusted_mutual_info_score(flag, y_pred, average_method='arithmetic'))

# palette
def cal_col(label):
    new_col = []
    colo_tar = ['r', 'g', 'b', 'y', 'k', 'c', 'gray']
    for i in range(len(label)):
        new_col.append(colo_tar[int(label[i])])
    return new_col