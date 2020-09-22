import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

m = 2
# cols to use for clustering
cols = ['NoFaceContact', 'Risk', 'Sick']


def find_starting_clusters(df, k):
    sample_df = df.drop_duplicates()
    starting_pts = sample_df.sample(n=k, random_state=1)
    starting_pts = starting_pts + 0.01
    return starting_pts


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def update_membership(df, k, cluster_df):
    for i in range(0, k):
        col_name = 'cluster ' + str(i)

        df[col_name] = 1.0 / (df['X'] - cluster_df[i][0])

    return df


def dist(list1, list2):
    val = 0
    for i in list1:
        val += (list1[i] - list2[i]) ** 2


def calc_mem(df, cluster_cen, k):
    tmp_list = []
    for i in range(0, k):
        rand = 0
        col_name = 'mem' + str(i)

        for n in range(0, len(cols)):
            rand += rand + (df[cols[n]] - cluster_cen[i][n]) ** 2
        tmp_list.append(rand)
    # mem = 1 / ((tmp_list[0] / ((tmp_list[1]) ** (2 / (m - 1)))) + (tmp_list[0] / ((tmp_list[2]) ** (2 / (m - 1)))))

    for p in range(0, k):
        col_name = 'mem' + str(p)
        dem = 0
        for q in range(0, k):
            dem = dem + (tmp_list[p] / (tmp_list[q])) ** (2 / (m - 1))

        df[col_name] = 1 / dem


def recenter_clusters(df, k):
    # cluster = df.groupby('cluster_mem').mean()
    cluster_cen = []
    points = df[cols].values.tolist()
    for p in range(0, k):
        name = 'mem' + str(p)
        temp = df[cols].multiply(df[name], axis='index')
        test = temp.sum()
        test2 = df[name].sum().tolist()
        cluster = (test / test2).tolist()
        cluster_cen.append(cluster)
    return cluster_cen
    # for q in points:
    # temp = df['mem'+str(p)].values.tolist()
    # return cluster_cen


def read_data(file):
    df = pd.read_csv(file)

    drop_col = (df.columns[~df.columns.isin(cols)].tolist())
    df.drop(drop_col, axis=1, inplace=True)
    df.fillna(0, inplace=True)
    # df = pd.DataFrame([[0,1,2,],[1,2,1],[0,0,0]], columns =['NoFaceContact', 'Risk','Sick'])
    return df


def plotly_chart(df):
    fig = px.scatter_3d(df, x='NoFaceContact', y='Risk', z='Sick',
                        color='mem0', symbol='type')

    fig.show()


def fuzzy_c_means(k):
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
        cluster_cols.append(1)

    cluster_list = find_starting_clusters(data_frame, k).values.tolist()
    print(cluster_list)

    while con:
        i += 1
        old_set = cluster_list
        calc_mem(data_frame, cluster_list, k)
        cluster_list = recenter_clusters(data_frame, k)
        new_set = cluster_list

        if i > max_iter or old_set == new_set:
            print(i)
            con = False

    zeroes = []
    zeroes.extend(cluster_cols)
    zeroes.append('Center')
    for clust in cluster_list:
        clust.extend(zeroes)
    # k_means_chart(data_frame)
    data_frame['type'] = 'cluster'
    test = list(data_frame.columns)
    temp = pd.DataFrame(cluster_list, columns=list(data_frame.columns))
    data_frame = data_frame.append(temp)

    plotly_chart(data_frame)


fuzzy_c_means(2)
