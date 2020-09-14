import pandas as pd
import math
import numpy as np

col = ['X', 'Y']


def find_starting_clusters(df, k):
    return df.sample(n=k)


def update_membership(df, k, cluster_df):
    for i in range(0, k):
        col_name = 'cluster ' + str(i)

        df[col_name] = 1.0 / (df['X'] - cluster_df[i][0])

    return df


def dist(list1, list2):
    val = 0
    for i in list1:
        val += (list1[i] - list2[i]) ** 2


def k_means(df, cluster_cen):
    for i in range(0, k):
        rand = 0
        col_name = 'k cluster ' + str(i)
        print(cluster_cen)

        for n in range(0, 2):
            rand += rand + (df[col[n]] - cluster_cen[i][n]) ** 2

        df[col_name] = rand


def classify_k_means(df):
    df['cluster_mem'] = df[['k cluster 0', 'k cluster 1']].idxmin(axis=1)
    # df[['X','Y']]


def recenter_clusters(df):
    cluster = df.groupby('cluster_mem').mean()
    cluster_cen = cluster[['X', 'Y']].values.tolist()
    return cluster_cen

def read_data(file):
    df = pd.read_csv(file)
    col = df.columns

sample = {'X': [1, 2, 4, 7], 'Y': [3, 5, 8, 9]}
data_frame = pd.DataFrame(sample, columns=['X', 'Y'])
k = 2

con = True
max_iter = 5000
i = 0
old_set = set()
new_set = set()

cluster_list = find_starting_clusters(data_frame, k).values.tolist()
print(cluster_list)
while con:
    i += 1
    old_set = cluster_list
    k_means(data_frame, cluster_list)
    classify_k_means(data_frame)
    cluster_list = recenter_clusters(data_frame)
    new_set = cluster_list
    print(data_frame)
    if i > max_iter or old_set == new_set:
        con = False
# print(cluster_list)
# print(update_membership(df,2,cluster_list))
