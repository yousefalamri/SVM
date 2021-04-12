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

def construct_objective_matrix():
    K = np.ndarray([N, N])
    for i in range(N):
        for j in range(N):
            K[i, j] = (train_data[i][-1]) * (train_data[j][-1]) * np.inner(train_data[i][0:ftr_len],train_data[j][0:ftr_len])
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
    bd = (0, C)
    bds = tuple([bd for i in range(N)])
    x0 = np.zeros(N)
    cons = {'type': 'eq', 'fun': constraint}
    sol = minimize(objective_function, x0, method='SLSQP', bounds=bds, constraints=cons)
    return [sol.fun, sol.x]


############################# SVM dual prediction ################################

# get w vector
def get_weights(dual_x):
    lenn = len(dual_x)
    ll = []
    for i in range(lenn):
        ll.append(dual_x[i] * train_data[i][-1] * np.asarray(train_data[i][0: ftr_len]))
    return sum(ll)


# counting the number of errors
def error_counter(prediction, actual):
    error_count = 0
    input_length = len(prediction)
    for i in range(input_length):
        if prediction[i] != actual[i]:
            error_count = error_count + 1
    return error_count / input_length

def calculate_error(w, input_data):
    prediction_vector = [];
    for i in range(len(input_data)):
        prediction_vector.append(sgn(np.inner(input_data[i][0:len(input_data[0]) - 1], w)))
    actual_labels = [elem[-1] for elem in input_data]
    error_per = error_counter(prediction_vector, actual_labels) * 100.0
    return error_per


def run_dual_SVM(C):
    [optimal_fun, optimal_var] = find_min(C)
    w = get_weights(optimal_var)
    training_error = calculate_error(w, train_data)
    training_test = calculate_error(w, test_data)
    print('weight vector=', w)
    print('training error=', training_error)
    print('testing error=', training_test)


############################# Run dual SVM ################################
Mat = construct_objective_matrix()
C_vals = [100 / 873, 500 / 873, 700 / 873]
for thisC in C_vals:
    run_dual_SVM(thisC)

