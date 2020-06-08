from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd


if __name__ == '__main__':
    # load data
    df = pd.read_table('.\data\Vertebral_column\column_2C_weka.txt', sep=',', header=None)
    data = df.iloc[:, 0:6]
    class_labels = df.iloc[:, 6]

    # --------------- Dimension Reduction --------------- #
    # Dimension Reduction: t-SNE
    data_tsne = TSNE(n_components=2, init='pca', random_state=0).fit_transform(data)
    dtsne_0 = data_tsne[class_labels == 'Normal']
    dtsne_1 = data_tsne[class_labels == 'Abnormal']

    # Dimension Reduction: PCA
    data_pca = PCA(n_components=6).fit_transform(data)
    dpca_0 = data_pca[class_labels == 'Normal']
    dpca_1 = data_pca[class_labels == 'Abnormal']

    # plot
    plt.subplot(221)
    plt.scatter(dtsne_0[:, 0], dtsne_0[:, 1], c='', s=40, marker='^', edgecolors='k')
    plt.scatter(dtsne_1[:, 0], dtsne_1[:, 1], c='k', s=80, marker='.', edgecolors='')
    plt.axis('off')

    plt.subplot(222)
    plt.xlim(-1e-8, 1e-8)
    plt.scatter(dpca_0[:, 5], dpca_0[:, 1], c='', s=40, marker='^', edgecolors='k')
    plt.scatter(dpca_1[:, 5], dpca_1[:, 1], c='k', s=80, marker='.', edgecolors='')
    plt.axis('off')

    plt.subplot(223)
    sns.kdeplot(dtsne_0[:, 0], shade=False, color='k', linestyle='--', linewidth=1.2)
    sns.kdeplot(dtsne_1[:, 0], shade=False, color='k', linestyle='-', linewidth=1.6)
    plt.axis('off')

    plt.subplot(224)
    sns.kdeplot(dpca_0[:, 5], shade=False, color='k', linestyle='--', linewidth=1.2)
    sns.kdeplot(dpca_1[:, 5], shade=False, color='k', linestyle='-', linewidth=1.6)
    plt.axis('off')

    # plt.savefig('results/Vertebral_column/dimension_reduction.png', dpi=512)
    plt.show()

