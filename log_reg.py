import numpy as np
import math
import pandas as pd

cols = ['NoFaceContact', 'Risk', 'Sick']


def read_data(file):
    '''
    load csv into pandas, and drop null values and noise
    '''
    df = pd.read_csv(file)

    # drop_col = (df.columns[~df.columns.isin(cols)].tolist())
    df = df.loc[df['NoFaceContact'] < 9]
    # df.drop(drop_col, axis=1, inplace=True)
    df.dropna(0, inplace=True)
    return df


def gradient_descent(f, training_pts, labels, alpha=1, ep=0.005, reg=0):
    continue_flag = True
    theta = np.full(len(training_pts.columns), 1)
    iteration = 0
    deriv_vals = []

    while continue_flag:
        col = 0
        for column_name, column_val in training_pts.iteritems():
            deriv = 0
            for i in range(0, len(column_val)):
                val = f(training_pts.loc[i].tolist(), theta)
                deriv += (val - labels[i]) * column_val[i]

            learning_rate = alpha / len(training_pts)
            regularized_term = (float(reg) / len(training_pts) * theta[col])

            theta[col] = theta[col] - (learning_rate * deriv) + regularized_term

            deriv_vals.append(abs(deriv))
            col += 1

        iteration += 1

        total_cost = cost_function(training_pts, labels, theta)
        if iteration == 200 or max(deriv_vals) < ep:
            continue_flag = False

        deriv_vals.clear()
    return theta


def com_confusion_matrix(data_frame, col):
    result = data_frame.groupby(['Flu', 'Predicted'])[col].count()
    try:
        tn = result.loc[(0.0, 0)]
    except KeyError:
        tn = 0
    try:
        tp = result.loc[(1.0, 1)]
    except KeyError:
        tp = 0

    try:
        fp = result.loc[(0.0, 1)]
    except KeyError:
        fp = 0

    try:
        fn = result.loc[(1.0, 0)]
    except KeyError:
        fn = 0

    try:
        precision = float(tp) / (tp + fp)
    except ZeroDivisionError:
        precision = 0
    recall = float(tp) / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)
    # print('TN' + str(tn))
    # print('TP' + str(tp))
    # print('FP' + str(fp))
    # print('FN' + str(fn))
    # print('Precision' + str(precision))
    # print('Recall' + str(recall))
    return f_score


def cost_function(training_pts, labels, theta):
    total = 0
    for i in range(0, len(training_pts)):
        try:
            left = labels[i] * math.log(fun(training_pts.loc[i], theta))
            right = (1 - labels[i]) * math.log(1 - fun(training_pts.loc[i], theta))
            total += left + right
        except:
            print(training_pts.loc[i])
            print(fun(training_pts.loc[i], theta))
            print('undefined')
        # log is undefined so make total infinite
    return (-1 / len(training_pts)) * total


def classify(testing_pts, theta):
    labels = []
    for i in range(0, len(testing_pts)):
        val = fun(testing_pts.loc[i], theta)
        if val > 0.5:
            labels.append(1)
        else:
            labels.append(0)
    return labels


def fun(x, theta):
    val = np.dot(x, theta)
    val *= -1
    return 1 / (1 + math.exp(val))


def forward_selection(df):
    f_scores_list = []
    selected_columns = []
    options = list(df.columns)

    for i in options:
        theta = gradient_descent(fun, df[[i]], df['Flu'].tolist())
        labels = classify(df[[i]], theta)
        data = df.copy()
        data['Predicted'] = labels
        f_score = com_confusion_matrix(data, i)
        f_scores_list.append(f_score)
    idx = f_scores_list.index(max(f_scores_list))
    selected_columns.append(options.pop(idx))
    print('')


file = r"C:\Users\Anthony\Downloads\Assignment1_Data.csv"
data_frame = read_data(file)
data_frame = data_frame.reset_index(drop=True)


# question 1
def question1():
    data = data_frame[['Risk', 'Flu']]
    theta = gradient_descent(fun, data['Risk'].tolist(), data['Flu'].tolist())
    labels = classify(data['Risk'].tolist(), theta)
    data['Predicted'] = labels


def question2():
    data = data_frame[['Risk', 'NoFaceContact', 'Flu']]
    data = data.reset_index(drop=True)
    theta = gradient_descent(fun, data[['Risk', 'NoFaceContact']], data['Flu'].tolist())
    print(theta)
    labels = classify(data[['Risk', 'NoFaceContact']], theta)
    data['Predicted'] = labels
    com_confusion_matrix(data)


def question3():
    data = data_frame[['Risk', 'NoFaceContact', 'Flu']]
    data = data.reset_index(drop=True)
    theta = gradient_descent(fun, data[['Risk', 'NoFaceContact']], data['Flu'].tolist(), reg=1)
    print(theta)
    labels = classify(data[['Risk', 'NoFaceContact']], theta)
    data['Predicted'] = labels
    com_confusion_matrix(data)


temp_list = ['HndWshQual', 'SociDist', 'NoFaceContact', 'Flu']
forward_selection(data_frame[temp_list])
