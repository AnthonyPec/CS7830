import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import dunns_index as di


class NotConvergeError(Exception):
    """K means algo failed to converge"""
    pass


# cols to use for clustering
cols = ['NoFaceContact', 'Risk', 'Sick','HandSanit']


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def find_starting_clusters(df, k):
    sample_df = df.drop_duplicates()
    starting_pts = sample_df.sample(n=k)
    return starting_pts


def update_membership(df, k, cluster_df):
    for i in range(0, k):
        col_name = 'cluster ' + str(i)

        df[col_name] = 1.0 / (df['X'] - cluster_df[i][0])

    return df


def dist(list1, list2):
    val = 0
    for i in list1:
        val += (list1[i] - list2[i]) ** 2


def calc_dist(df, cluster_cen, k):
    for i in range(0, k):
        rand = 0
        col_name = 'k cluster ' + str(i)

        for n in range(0, len(cols)):
            try:
                rand += rand + (df[cols[n]] - cluster_cen[i][n]) ** 2
            except:
                print('here')

        df[col_name] = rand


def classify_k_means(df, cluster_cols,k):
    df['cluster_mem'] = df[cluster_cols].idxmin(axis=1)

    count = df['cluster_mem'].unique()

    # cluster was lost, k means failed to converge
    if len(count) < int(k):
        raise NotConvergeError

def recenter_clusters(df,k):
    cluster = df.groupby('cluster_mem').mean()
    cluster_cen = cluster[cols].values.tolist()
    return cluster_cen


def read_data(file):
    df = pd.read_csv(file)

    drop_col = (df.columns[~df.columns.isin(cols)].tolist())
    df.drop(drop_col, axis=1, inplace=True)
    df.dropna(0, inplace=True)
    return df


def k_means_chart(df):
    colmap = {'k cluster 0': 'r', 'k cluster 1': 'g', 'k cluster 2': 'b', 'k cluster 3': 'purple'}
    # colors = ['blue', 'green', 'red']
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection="3d")

    labels = list(set(df['cluster_mem'].tolist()))
    # centroids = cluster_list
    # colors = map(lambda x: colmap[x], labels)

    # ax.scatter3D(df['NoFaceContact'], df['Risk'], df['Sick'], color=list(colors), alpha=0.5)

    for i in range(len(df)):
        ax.scatter3D(df.iloc[i]['NoFaceContact'], df.iloc[i]['Risk'], df.iloc[i]['Sick'],
                     c=colmap[(df.iloc[i]['cluster_mem'])])

    plt.show()


def plotly_chart(df):
    fig = px.scatter_3d(df, x='NoFaceContact', y='Risk', z='Sick',
                        color='cluster_mem', symbol='type')
    fig.show()


def k_means(k):
    k = k

    con = True
    max_iter = 5000
    i = 0
    old_set = set()
    new_set = set()
    file = r"C:\Users\Anthony\Downloads\Assignment1_Data.csv"
    data_frame = read_data(file)
    data_frame = normalize(data_frame)
    cluster_cols = []
    for i in range(0, k):
        cluster_cols.append('k cluster ' + str(i))

    cluster_list = find_starting_clusters(data_frame, k).values.tolist()

    while con:
        i += 1
        old_set = cluster_list
        calc_dist(data_frame, cluster_list, k)
        classify_k_means(data_frame, cluster_cols,k)
        cluster_list = recenter_clusters(data_frame,k)
        new_set = cluster_list
        if i > max_iter or old_set == new_set:
            #print(i)
            con = False

    zeroes = []
    zeroes.extend(cluster_cols)
    zeroes.append('Center')
    zeroes.append('Center')
    for clust in cluster_list:
        clust.extend(zeroes)
    # k_means_chart(data_frame)
    data_frame['type'] = 'cluster'
    test = list(data_frame.columns)
    temp = pd.DataFrame(cluster_list, columns=list(data_frame.columns))
    data_frame = data_frame.append(temp)
    #plotly_chart(data_frame)
    return data_frame



def avg_dunn_index():
    avg_dunn_index = np.zeros(11)
    for index in range(0, 5):
        for k in range(2, 11):
            data_frame = k_means(k)
            dunn_index = di.calc_dunn_index(data_frame, k, cols)
            avg_dunn_index[k] += dunn_index
    avg_dunn_index = avg_dunn_index / 20
    for i in range(2,11):
        print('K = ' + str(i) + ' has a dunn index of ' + str(avg_dunn_index[i]))


avg_dunn_index()
#for k in range(2, 10):
#    data_frame = k_means(k)
# for cluster_num in range(2, 10):
# k_means(cluster_num)
