import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import math
import statistics

############################# supporting functions for Perceptron ################################
# Load train.csv and test.csv
with open('bank-note/train.csv') as f:
    training_data = [];
    for line in f:
        terms = line.strip().split(',')
        training_data.append(terms)

with open('bank-note/test.csv') as f:
    testing_data = [];
    for line in f:
        terms = line.strip().split(',')
        testing_data.append(terms)


def convert_to_float(input_data):
    for elem in input_data:
        for i in range(len(input_data[0])):
            elem[i] = float(elem[i])
    return input_data

# augment feature vector
def augment_feature_vector(input_data):
    labels = [elem[-1] for elem in input_data]
    data_list = input_data
    for i in range(len(input_data)):
        data_list[i][-1] = 1.0
    for i in range(len(input_data)):
        data_list[i].append(labels[i])
    return data_list


# convert (0,1) labels t0 (-1,1)
def convert_to_pm_one(input_data):
    new_list = input_data
    for i in range(len(input_data)):
        if new_list[i][-1] != 1.0:
            new_list[i][-1] = -1.0
    return new_list

# pre-process data
training_data = convert_to_float(training_data)
testing_data = convert_to_float(testing_data)

train_data = augment_feature_vector(convert_to_pm_one(training_data))
test_data = augment_feature_vector(convert_to_pm_one(testing_data))

N = len(train_data)

# scheduling learning rate for part a
def schd_lrn_rate_a(t, gamma_0, d):
    return gamma_0 / (1 + gamma_0 * t / d)

# scheduling learning rate for part b
def schd_lrn_rate_b(t, gamma_0):
    return gamma_0 / (1 + t)


################################## Sub-gradient Algorithm ##################################

def run_subgradient(w, input_sample, iter, scheduling_type, C, gamma_0, d):
    W = list(np.zeros(len(input_sample) - 1))
    w_0 = w[0:len(w) - 1];
    w_0.append(0)
    ww_0 = w_0
    if scheduling_type == 1:
        term_1 = 1 - schd_lrn_rate_a(iter, gamma_0, d)
        term_2 = schd_lrn_rate_a(iter, gamma_0, d)
        term_3 = term_2 * C * N * input_sample[-1]
        if input_sample[-1] * np.inner(input_sample[0:len(input_sample) - 1], w) <= 1:
            new_w_1 = [x * term_1 for x in ww_0]
            new_w_2 = [x * term_3 for x in input_sample[0:len(input_sample) - 1]]
            W = [new_w_1[i] + new_w_2[i] for i in range(len(new_w_1))]
        else:
            W = [x * term_1 for x in ww_0]
    if scheduling_type == 2:
        term_1 = 1 - schd_lrn_rate_b(iter, gamma_0)
        term_2 = schd_lrn_rate_b(iter, gamma_0)
        term_3 = term_2 * C * N * input_sample[-1]
        if input_sample[-1] * np.inner(input_sample[0:len(input_sample) - 1], w) <= 1:
            new_w_1 = [x * term_1 for x in ww_0]
            new_w_2 = [x * term_3 for x in input_sample[0:len(input_sample) - 1]]
            W = [new_w_1[i] + new_w_2[i] for i in range(len(new_w_1))]
        else:
            W = [x * term_1 for x in ww_0]
    return W

################################## SVM Algorithm ##################################

# sign function
def sgn(input):
    sign = 0
    if input > 0:
        sign = 1
    else:
        sign = -1
    return sign

# loss function
def calculate_loss(w, C, input_data):
    loss_vector = [];
    for i in range(N):
        loss_vector.append(max(0, 1 - input_data[i][-1] * np.inner(w, input_data[i][0:len(input_data[0]) - 1])))
    loss = 0.5 * np.linalg.norm(w) ** 2 + C * sum(loss_vector)
    return loss


# Perform one iteration of SVM
def SVM(w, iter, shfl_vec, input_data, C, scheduling_type, gamma_0, d):
    loss_vector = [];
    for i in range(N):
        w = run_subgradient(w, input_data[shfl_vec[i]], iter, scheduling_type, C, gamma_0, d)
        loss_vector.append(calculate_loss(w, C, input_data))
        iter = iter + 1
    return [w, iter, loss_vector]

# Perform T iterations of SVM
def repeat_SVM(w, T, train_data, C, scheduling_type, gamma_0, d):
    iter = 1
    loss_vector = []
    #    ct = 0
    for i in range(T):
        shfl_vec = np.random.permutation(N)
        [w, iter, thisloss] = SVM(w, iter, shfl_vec, train_data, C, scheduling_type, gamma_0, d)
        loss_vector.extend(thisloss)
    #        ct = ct + 1
    #        print('# epoch=',ct)
    return [w, loss_vector]

# counting the number of errors
def error_counter(prediction, actual):
    error_count = 0
    input_length = len(prediction)
    for i in range(input_length):
        if prediction[i] != actual[i]:
            error_count = error_count + 1
    return error_count / input_length

# do prediction and calculate error percentage
def calculate_error(w, input_data):
    prediction_vector = [];
    for i in range(len(input_data)):
        prediction_vector.append(sgn(np.inner(input_data[i][0:len(input_data[0]) - 1], w)))
    actual_labels = [elem[-1] for elem in input_data]
    error_per = error_counter(prediction_vector, actual_labels) * 100.0
    return error_per

# main function that runs SVM
def run_SVM(scheduling_type, T, gamma_0, d):
    hyper_Cs = [x / 873 for x in [100, 500, 700]]  # hyper parameter
    for C in hyper_Cs:
        w = list(np.zeros(len(train_data[0]) - 1))
        [weights, loss] = repeat_SVM(w, T, train_data, C, scheduling_type, gamma_0, d)
        print('Learned weight vector:', weights)
        #print('loss',loss)
        training_error = calculate_error(weights, train_data)
        testing_error = calculate_error(weights, test_data)
        print('Training error:', training_error)
        print('Testing error:', testing_error)
    plt.plot(loss)
    plt.show()




################################## Run SVM Algorithm ##################################
# scheduling learning rate: choose 1 for part a, 2 for part b
scheduling_type = 1
# number of epochs
T = 100
# initial learning rate
gamma_0 = 2
# the d term in the learning rate update
d = 1
# run SVM
run_SVM(scheduling_type, T, gamma_0, d)

