import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import dunns_index as di

# parameter to control how fuzzy cluster will be
m = 2

# cols to use for clustering
cols = ['NoFaceContact', 'Risk', 'Sick']



def find_starting_clusters(df, k):
    '''
    find starting clusters for fuzzy c-means
    '''
    sample_df = df.drop_duplicates()
    starting_pts = sample_df.sample(n=k)

    starting_pts = starting_pts + 0.01

    return starting_pts


def standardize_data(df):
    '''
    standardize each column besides Risk since it is already standardize
    '''
    result = df.copy()
    for feature_name in df.columns:
        # Risk is already normalized
        if feature_name == 'Risk':
            continue
        result[feature_name] = (df[feature_name] - df[feature_name].mean()) / df[feature_name].std()
    return result


def calc_mem(df, cluster_cen, k):
    '''
    calculate membership of each entry in dataframe for each cluster
    '''
    tmp_list = []
    for i in range(0, k):
        rand = 0
        col_name = 'cluster ' + str(i)

        for n in range(0, len(cols)):
            rand += rand + (df[cols[n]] - cluster_cen[i][n]) ** 2
        tmp_list.append(rand)
    # mem = 1 / ((tmp_list[0] / ((tmp_list[1]) ** (2 / (m - 1)))) + (tmp_list[0] / ((tmp_list[2]) ** (2 / (m - 1)))))

    for p in range(0, k):
        col_name = 'cluster ' + str(p)
        dem = 0
        for q in range(0, k):
            dem = dem + (tmp_list[p] / (tmp_list[q])) ** (2 / (m - 1))

        df[col_name] = 1 / dem


def recenter_clusters(df, k):
    '''
    calculate new cluster centers
    '''
    # cluster = df.groupby('cluster_mem').mean()
    cluster_cen = []
    points = df[cols].values.tolist()
    for p in range(0, k):
        name = 'cluster ' + str(p)

        temp = df[cols].multiply(df[name] ** (2), axis='index')
        test = temp.sum()
        test2 = (df[name] ** 2).sum().tolist()
        cluster = (test / test2).tolist()
        cluster_cen.append(cluster)
    return cluster_cen
    # for q in points:
    # temp = df['mem'+str(p)].values.tolist()
    # return cluster_cen


def read_data(file):
    '''
    load csv into pandas, and drop null values and noise
    '''
    df = pd.read_csv(file)
    drop_col = (df.columns[~df.columns.isin(cols)].tolist())
    df.drop(drop_col, axis=1, inplace=True)
    df.fillna(0, inplace=True)
    df = df.loc[df['NoFaceContact'] < 9]
    # df = pd.DataFrame([[0,1,2,],[1,2,1],[0,0,0]], columns =['NoFaceContact', 'Risk','Sick'])
    return df


def plotly_chart(df):
    '''
    plot results
    '''
    fig = px.scatter_3d(df, x='NoFaceContact', y='Risk', z='Sick',
                        color='cluster 0', symbol='type')

    fig.show()


def harden_cluster(df, cluster_cols):
    '''
    assign cluster to each row based on highest memebership
    '''
    df['cluster_mem'] = df[cluster_cols].idxmax(axis=1)


def fuzzy_c_means(k):
    '''
    fuzzy c-means algo, find starting cluster then calc mem , then recalculate centers, repeat this
    until max iterations are meet or we dont decrease obj function
    '''
    k = k

    con = True

    max_iter = 500
    i = 0
    old_set = set()
    new_set = set()
    file = r"C:\Users\Anthony\Downloads\Assignment1_Data.csv"
    data_frame = read_data(file)
    data_frame = standardize_data(data_frame)
    cluster_cols = []
    for i in range(0, k):
        cluster_cols.append('cluster ' + str(i))

    cluster_list = find_starting_clusters(data_frame, k).values.tolist()
    min_sum = 999999
    while con:
        i += 1
        calc_mem(data_frame, cluster_list, k)
        cluster_list = recenter_clusters(data_frame, k)
        sum = obj_fun(data_frame, cluster_list, k)
        if sum < min_sum:
            min_sum = sum
        else:
            print(i)
            con = False

        if i > max_iter:
            print(i)
            con = False

    zeroes = []
    for clust in cluster_cols:
        zeroes.append(.5)
    zeroes.extend(['Center', 'Cluster'])
    for clust in cluster_list:
        clust.extend(zeroes)
    # k_means_chart(data_frame)
    data_frame['type'] = 'cluster'
    harden_cluster(data_frame, cluster_cols)
    temp = pd.DataFrame(cluster_list, columns=list(data_frame.columns))
    data_frame = data_frame.append(temp)
    plotly_chart(data_frame)
    return data_frame, cluster_list


def obj_fun(df, cluster_list, k):
    '''
    objective function we are trying to minimize for fuzzy c means
    '''
    tmp_list = []
    sum = 0
    for i in range(0, k):
        rand = 0
        col_name = 'cluster ' + str(i)

        for n in range(0, len(cols)):
            rand += rand + (df[cols[n]] - cluster_list[i][n]) ** 2
        rand = rand ** (1 / 2)
        sum += df[col_name].multiply(rand).sum()
    return sum


def run_multi_dunn_index(start_clust, end_clust):
    '''
    run multi iterations of fuzzy c means and calc avg DI and highest DI
    '''
    avg_dunn_index = np.zeros(11)
    high_dunn_index = np.zeros(11)
    for index in range(0, 50):
        for k in range(start_clust, end_clust + 1):
            data_frame, clust_list = fuzzy_c_means(k)
            dunn_index = di.calc_dunn_index(data_frame, k, cols)
            avg_dunn_index[k] += dunn_index
            if dunn_index > high_dunn_index[k]:
                high_dunn_index[k] = dunn_index
    avg_dunn_index = avg_dunn_index / 50
    for i in range(start_clust, end_clust + 1):
        print('K = ' + str(i) + ' has an average dunn index of ' + str(avg_dunn_index[i]))
    for i in range(start_clust, end_clust + 1):
        print('K = ' + str(i) + ' highest dunn index of ' + str(high_dunn_index[i]))



# run fuzzy c-means with k = 3 and print centers
data_frame, clust_list = fuzzy_c_means(3)
for i in clust_list:
    print(i[0:3])

# run dunn index on with k = 3
run_multi_dunn_index(3, 3)
