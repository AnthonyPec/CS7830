import pandas as pd
import numpy as np
import plotly.express as px
import dunns_index as di

# cols to use for clustering
cols = ['NoFaceContact', 'Risk', 'Sick']



def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def standardize_data(df):
    result = df.copy()
    for feature_name in df.columns:
        # Risk is already normalized
        if feature_name == 'Risk':
            continue
        result[feature_name] = (df[feature_name] - df[feature_name].mean()) / df[feature_name].std()
    return result


def find_starting_clusters(df, k):
    sample_df = df.drop_duplicates()
    starting_pts = sample_df.sample(n=k)
    return starting_pts


def dist(list1, list2):
    val = 0
    for i in list1:
        val += (list1[i] - list2[i]) ** 2


def calc_dist(df, cluster_cen, k):
    '''
    calc distance from each point to each cluster center
    '''
    for i in range(0, k):
        rand = 0
        col_name = 'cluster ' + str(i)

        for n in range(0, len(cols)):
            try:
                rand += rand + (df[cols[n]] - cluster_cen[i][n]) ** 2
            except:
                print('here')

        df[col_name] = rand ** (1 / 2)


def classify_k_means(df, cluster_cols, k):
    '''
    classify each row of dataframe by assigning it to cluster closest to it
    '''
    df['cluster_mem'] = df[cluster_cols].idxmin(axis=1)

    count = df['cluster_mem'].unique()

    # some clusters are empty need to replace with random point
    if len(count) < int(k):
        empty_sets = set(cluster_cols) - set(count)
        sample_df = df.drop_duplicates()
        starting_pts = sample_df.sample(n=len(empty_sets))
        test = starting_pts.index.values
        for i in range(0, len(empty_sets)):
            df.loc[test[i], 'cluster_mem'] = empty_sets.pop()


def recenter_clusters(df, k):
    '''
    recalc cluster centers based on what points are in cluster
    '''
    cluster = df.groupby('cluster_mem').mean()
    cluster_cen = cluster[cols].values.tolist()
    return cluster_cen


def read_data(file):
    '''
    load csv into pandas, and drop null values and noise
    '''
    df = pd.read_csv(file)

    drop_col = (df.columns[~df.columns.isin(cols)].tolist())
    df = df.loc[df['NoFaceContact'] < 9]
    df.drop(drop_col, axis=1, inplace=True)
    df.dropna(0, inplace=True)
    return df



def plotly_chart(df):
    '''
    plot results
    '''
    fig = px.scatter_3d(df, x='NoFaceContact', y='Risk', z='Sick',
                        color='cluster_mem', symbol='type')
    fig.show()


def k_means(k, plot = True):
    k = k

    con = True
    max_iter = 5000
    i = 0
    old_set = set()
    new_set = set()
    file = r"C:\Users\Anthony\Downloads\Assignment1_Data.csv"
    data_frame = read_data(file)

    data_frame = standardize_data(data_frame)
    #data_frame = normalize(data_frame)


    cluster_cols = []
    for i in range(0, k):
        cluster_cols.append('cluster ' + str(i))

    cluster_list = find_starting_clusters(data_frame, k).values.tolist()
    while con:
        i += 1
        old_set = cluster_list
        calc_dist(data_frame, cluster_list, k)
        classify_k_means(data_frame, cluster_cols, k)
        cluster_list = recenter_clusters(data_frame, k)
        new_set = cluster_list
        if i > max_iter or old_set == new_set:
            # print(i)
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
    #print(di.calc_dunn_index(data_frame, k, cols))
    return data_frame, cluster_list


def run_multi_dunn_index(start_clust, end_clust):
    #run multi iterations of k means and calc avg DI and highest DI
    avg_dunn_index = np.zeros(11)
    high_dunn_index = np.zeros(11)
    for index in range(0, 50):
        for k in range(start_clust, end_clust+1):
            data_frame, clust_list = k_means(k,False)
            dunn_index = di.calc_dunn_index(data_frame, k, cols)
            avg_dunn_index[k] += dunn_index
            if dunn_index > high_dunn_index[k]:
                high_dunn_index[k] = dunn_index
    avg_dunn_index = avg_dunn_index / 50
    for i in range(start_clust, end_clust+1):
        print('K = ' + str(i) + ' has an average dunn index of ' + str(avg_dunn_index[i]))
    for i in range(start_clust,end_clust+1):
        print('K = ' + str(i) + ' highest dunn index of ' + str(high_dunn_index[i]))


#run dunn index for k = 2 to 10

# run k means for k =2 to k =10 and print cluster points and plot each
for k in range(2, 10):
    data_frame,clust_list = k_means(k)
    for i in clust_list:
        print(i[0:3])

# run DI on k = 2 to 10
run_multi_dunn_index(2,10)
