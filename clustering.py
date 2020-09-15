import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

# cols to use for clustering
cols = ['NoFaceContact', 'Risk', 'Sick']


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


def k_means(df, cluster_cen):
    for i in range(0, k):
        rand = 0
        col_name = 'k cluster ' + str(i)

        for n in range(0, len(cols)):
            rand += rand + (df[cols[n]] - cluster_cen[i][n]) ** 2

        df[col_name] = rand


def classify_k_means(df):
    df['cluster_mem'] = df[cluster_cols].idxmin(axis=1)


def recenter_clusters(df):
    cluster = df.groupby('cluster_mem').mean()
    cluster_cen = cluster[cols].values.tolist()
    return cluster_cen


def read_data(file):
    df = pd.read_csv(file)

    drop_col = (df.columns[~df.columns.isin(cols)].tolist())
    df.drop(drop_col, axis=1, inplace=True)
    df.fillna(0, inplace=True)
    return df


def k_means_chart(df):
    colmap = {'k cluster 0': 'r', 'k cluster 1': 'g', 'k cluster 2': 'b', 'k cluster 3': 'purple'}
    #colors = ['blue', 'green', 'red']
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection="3d")

    labels = list(set(df['cluster_mem'].tolist()))
    centroids = cluster_list
    #colors = map(lambda x: colmap[x], labels)

    #ax.scatter3D(df['NoFaceContact'], df['Risk'], df['Sick'], color=list(colors), alpha=0.5)

    for i in range(len(df)):
        ax.scatter3D(df.iloc[i]['NoFaceContact'], df.iloc[i]['Risk'], df.iloc[i]['Sick'], c=colmap[(df.iloc[i]['cluster_mem'])])

    plt.show()


# sample = {'X': [1, 2, 4, 7], 'Y': [3, 5, 8, 9]}
# data_frame = pd.DataFrame(sample, columns=['X', 'Y'])
k = 3

con = True
max_iter = 5000
i = 0
old_set = set()
new_set = set()
file = r"C:\Users\Anthony\Downloads\Assignment1_Data.csv"
data_frame = read_data(file)

cluster_cols = []
for i in range(0, k):
    cluster_cols.append('k cluster ' + str(i))

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
        print(i)
        con = False

k_means_chart(data_frame)

# print(cluster_list)
# print(update_membership(df,2,cluster_list))
