import math


def calc_dunn_index(df, num_cluster, col_names):
    max_diam = calc_diam(df, num_cluster, col_names)
    final_dist_set = set()
    for i in range(0,num_cluster):
        for j in range(i+1,num_cluster):
            name1 = 'k cluster ' + str(i)
            name2 = 'k cluster ' + str(j)
            c_i = df.loc[df['cluster_mem'] == name1][col_names].values.tolist()
            c_j = df.loc[df['cluster_mem'] == name2][col_names].values.tolist()
            dist_set = set()
            # not very efficient to find dist
            for m in c_i:
                for k in c_j:
                    rand = 0
                    for l in range(0, len(k)):
                        rand += (m[l] - k[l]) ** 2
                    dist_set.add(math.sqrt(rand))
            final_dist_set.add(min(dist_set))
    num = min(final_dist_set)
    return num / max_diam



def calc_diam(df, num_cluster, col_names):
    diam_set = set()
    for k in range(0, num_cluster):

        name = 'k cluster ' + str(k)
        temp = df.loc[df['cluster_mem'] == name]
        vals = temp[col_names].values.tolist()
        diam_set.add(calc_dist(vals))
    return max(diam_set)


def calc_dist(points_list):
    dist_set = set()
    # not very efficient to find dist
    for i in points_list:
        for k in points_list:
            rand = 0
            for l in range(0, len(k)):
                rand += (i[l] - k[l]) ** 2
            dist_set.add(math.sqrt(rand))
    return max(dist_set)
