import math
import numpy as np
from scipy.optimize import minimize

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

N = len(train_data)  # NO. of samples
ftr_len = len(train_data[0]) - 1  # sample dim
labels_vector = [row[-1] for row in train_data]

# sign function
def sgn(input):
    sign = 0
    if input > 0:
        sign = 1
    else:
        sign = -1
    return sign

############################# supporting functions for optimization ################################

# Gaussian kernel
def construct_kernel(xi, xj, gamma):
    xxi = np.asarray(xi)
    xxj = np.asarray(xj)
    return math.e ** (-np.linalg.norm(xxi - xxj) ** 2 / gamma)

def construct_objective_matrix(gamma):
    K = np.ndarray([N, N])
    for i in range(N):
        for j in range(N):
            K[i, j] = construct_kernel(train_data[i][0:ftr_len], train_data[j][0:ftr_len], gamma)
    return K

def objective_function(x):
    term1 = x.dot(Mat)
    term2 = term1.dot(x)
    term3 = -1 * sum(x)
    return 1/2 * term2 + term3

def constraint(x):
    return np.inner(x, np.asarray(labels_vector))


# minimization
def find_min(C):
    x0 = np.zeros(N)
    bd = (0, C)
    bds = tuple([bd for i in range(N)])
    cons = {'type': 'eq', 'fun': constraint}
    sol = minimize(objective_function, x0, method='SLSQP', bounds=bds, constraints=cons)
    return [sol.fun, sol.x]

############################# SVM dual prediction ################################

# counting the number of errors
def error_counter(prediction, actual):
    error_count = 0
    input_length = len(prediction)
    for i in range(input_length):
        if prediction[i] != actual[i]:
            error_count = error_count + 1
    return error_count / input_length


def calculate_error(optimal_x, data, gamma):
    actual_labels = [row[-1] for row in data]
    prediction_vector = [];
    for row in data:
        temp = []
        for i in range(len(optimal_x)):
            temp.append(optimal_x[i] * train_data[i][-1] * construct_kernel(train_data[i][0:ftr_len], row[0:ftr_len], gamma))
        predictions = sgn(sum(temp))
        prediction_vector.append(predictions)
    return error_counter(prediction_vector, actual_labels) * 100.0

# count the number of support vectors
def count_support(input_x):
    supp_vectors = []
    for i in range(len(input_x)):
        if input_x[i] != 0.0:
            supp_vectors.append(i)
    return [np.count_nonzero(input_x), set(supp_vectors)]


def run_dual_SVM(C):
    [optimal_func, optimal_var] = find_min(C)
    [count, indx] = count_support(optimal_var)
    print(count)
    print(indx)
    training_error = calculate_error(optimal_var, train_data, gamma)
    testing_error = calculate_error(optimal_var, test_data, gamma)
    print('Training error=', training_error)
    print('Testing error=', testing_error)
    return [count, indx]

############################# SVM dual prediction ################################


############################################ part b ###########################################

C_vals = [100 / 873, 500 / 873, 700 / 873]
gamma_vals = [0.1, 0.5, 1, 5, 100]

for C in C_vals:
    for gamma in gamma_vals:
        print('C=',C, 'gamma=', gamma)
        Mat = construct_objective_matrix(gamma)
        run_dual_SVM(C)

############################################ part c ###########################################

C = 500 / 873
supp_vector = []
for gamma in gamma_vals:
    print('C=', C, 'gamma=', gamma)
    Mat = construct_objective_matrix(gamma)
    [count, indx] = run_dual_SVM(C)
    supp_vector.append(indx)

overlap_vector = []
for k in range(len(gamma_vals) - 1):
    overlap_vector.append(len(supp_vector[k].intersection(supp_vector[k + 1])))
print('number of overlaps=', overlap_vector)

