import sys
import os
import numpy as np
from scipy import stats
import scipy.io as scio
import torch

from libsvm.svmutil import *


def mad(sequence, flag = 0):
    if flag == 0:
        return np.mean(np.abs(sequence - np.mean(sequence)))
    else:
        return np.median(np.abs(sequence - np.median(sequence)))

def xentrophy(sequence, bin_num = 10):
    min_val = np.min(sequence)
    max_val = np.max(sequence)
    bins = np.linspace(min_val, max_val, bin_num)
    result = 0
    for i in range(bin_num - 1):
        count = np.sum((sequence >= bins[i]) & (sequence < bins[i + 1]))
        if count > 0:
            p = count / len(sequence)
            result += (-p * np.log(p))
    return result


def load_x_y(file_path, mode="simple"):
    result_x, result_y = [], []
    mat = scio.loadmat(file_path)
    n, c, s = mat['accData'].shape
    if mode == "simple":
        for i in range(n):
            acc_vector = mat['accData'][i].reshape(c * s)
            gyr_vector = mat['gyrData'][i].reshape(c * s)
            vector = np.concatenate([acc_vector, gyr_vector])
            result_x.append(vector)
            result_y.append(mat['label'][i][0])
    else:
        for i in range(n):
            acc_vector = np.zeros(9 * c)
            gyr_vector = np.zeros(9 * c)
            for axis in range(c):
                acc_vector[9 * axis + 0] = np.mean(mat['accData'][i][axis])
                acc_vector[9 * axis + 1] = np.std(mat['accData'][i][axis])
                acc_vector[9 * axis + 2] = mad(mat['accData'][i][axis])
                acc_vector[9 * axis + 3] = mad(mat['accData'][i][axis], flag=1)
                acc_vector[9 * axis + 4] = np.max(mat['accData'][i][axis])
                acc_vector[9 * axis + 5] = np.min(mat['accData'][i][axis])
                acc_vector[9 * axis + 6] = stats.skew(mat['accData'][i][axis])
                acc_vector[9 * axis + 7] = stats.kurtosis(mat['accData'][i][axis])
                acc_vector[9 * axis + 8] = xentrophy(mat['accData'][i][axis])

                gyr_vector[9 * axis + 0] = np.mean(mat['gyrData'][i][axis])
                gyr_vector[9 * axis + 1] = np.std(mat['gyrData'][i][axis])
                gyr_vector[9 * axis + 2] = mad(mat['gyrData'][i][axis])
                gyr_vector[9 * axis + 3] = mad(mat['gyrData'][i][axis], flag=1)
                gyr_vector[9 * axis + 4] = np.max(mat['gyrData'][i][axis])
                gyr_vector[9 * axis + 5] = np.min(mat['gyrData'][i][axis])
                gyr_vector[9 * axis + 6] = stats.skew(mat['gyrData'][i][axis])
                gyr_vector[9 * axis + 7] = stats.kurtosis(mat['gyrData'][i][axis])
                gyr_vector[9 * axis + 8] = xentrophy(mat['gyrData'][i][axis])

            vector = np.concatenate([acc_vector, gyr_vector])
            result_x.append(vector)
            result_y.append(mat['label'][i][0])
    return result_x, result_y

def calc_precision_recall_f1(confusion, n_classes):
    precision = [0 for _ in range(n_classes)]
    recall = [0 for _ in range(n_classes)]
    f1 = [0 for _ in range(n_classes)]

    for i in range(n_classes):
        precision[i] = confusion[i][i] / np.sum(confusion[i, :]) if np.sum(confusion[i, :]) != 0 else 0
        recall[i] = confusion[i][i] / np.sum(confusion[:, i]) if np.sum(confusion[:, i]) != 0 else 0

    for i in range(n_classes):
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) != 0 else 0

    return precision, recall, f1


if __name__ == '__main__':
    datasource_path = os.path.join("/data/MaskReminder")
    method_name = "padding"
    strategy_name = "normal"
    seq_len = 128

    accuracy = []
    confusion = np.zeros((18, 18), dtype=np.int32)
    for i in range(5):
        train_x, train_y = load_x_y(os.path.join(datasource_path,
                                                 "%s-%s_%d-%d" % (method_name, strategy_name, i, seq_len),
                                                 "train.mat"), "statistic")
        train_x, train_y = np.array(train_x), np.array(train_y)
        model = svm_train(train_y, train_x)

        test_x, test_y = [], []
        temp_x, temp_y = load_x_y(os.path.join(datasource_path,
                                                 "%s-%s_%d-%d" % (method_name, strategy_name, i, seq_len),
                                                 "left_test.mat"), "statistic")
        test_x.extend(temp_x)
        test_y.extend(temp_y)
        temp_x, temp_y = load_x_y(os.path.join(datasource_path,
                                                 "%s-%s_%d-%d" % (method_name, strategy_name, i, seq_len),
                                                 "right_test.mat"), "statistic")
        test_x.extend(temp_x)
        test_y.extend(temp_y)
        p_labels, p_acc, p_vals = svm_predict(test_y, test_x, model)

        accuracy.append(p_acc[0])
        for pred, gt in zip(p_labels, test_y):
            confusion[int(pred)][int(gt)] += 1

    precision, recall, f1 = calc_precision_recall_f1(confusion, 18)
    print("mAccuracy: %.3f" % np.mean(accuracy))
    print("mPrecision: %.3f" % np.mean(precision))
    print("mRecall: %.3f" % np.mean(recall))
    print("mF1: %.3f" % np.mean(f1))