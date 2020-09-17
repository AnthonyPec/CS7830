import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

m = 4
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


def c_means(df, cluster_cen):
    tmp_list = []
    for i in range(0, k):
        rand = 0
        col_name = 'mem' + str(i)

        for n in range(0, len(cols)):
            rand += rand + (df[cols[n]] - cluster_cen[i][n]) ** 2
        tmp_list.append(rand)
    #mem = 1 / ((tmp_list[0] / ((tmp_list[1]) ** (2 / (m - 1)))) + (tmp_list[0] / ((tmp_list[2]) ** (2 / (m - 1)))))

    for p in range(0, k):
        col_name = 'mem' + str(p)
        dem = 0
        for q in range(0, k):
            dem = dem + (tmp_list[p] / (tmp_list[q])) ** (2 / (m - 1))

        df[col_name] = 1 / dem


def recenter_clusters(df):
    #cluster = df.groupby('cluster_mem').mean()
    #cluster_cen = cluster[cols].values.tolist()
    points = df[cols].values.tolist()
    for p in range(0,k):

        name = 'mem'+str(p)
        temp = df[cols].multiply(df[name],axis = 'index')
        print("")
       # for q in points:
        #temp = df['mem'+str(p)].values.tolist()
    #return cluster_cen


def read_data(file):
    df = pd.read_csv(file)

    drop_col = (df.columns[~df.columns.isin(cols)].tolist())
    df.drop(drop_col, axis=1, inplace=True)
    df.fillna(0, inplace=True)
    return df


def plotly_chart(df):
    fig = px.scatter_3d(df, x='NoFaceContact', y='Risk', z='Sick',
                        color='cluster_mem')
    fig.show()


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
    cluster_cols.append('mem' + str(i))

cluster_list = find_starting_clusters(data_frame, k).values.tolist()
print(cluster_list)
while con:
    i += 1
    old_set = cluster_list
    c_means(data_frame, cluster_list)
    cluster_list = recenter_clusters(data_frame)
    new_set = cluster_list
    print(data_frame)
    if i > max_iter or old_set == new_set:
        print(i)
        con = False

# k_means_chart(data_frame)
plotly_chart(data_frame)
# print(cluster_list)
# print(update_membership(df,2,cluster_list))
