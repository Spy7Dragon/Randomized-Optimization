import mlrose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sklearn.metrics import *

columns = [
    "annual_inc",
    "collections_12_mths_ex_med",
    "delinq_amnt",
    "delinq_2yrs",
    "dti",
    "fico_range_high",
    "fico_range_low",
    "home_ownership",
    "inq_last_6mths",
    "installment",
    "int_rate",
    "verification_status",
    "loan_amnt",
    "mths_since_last_major_derog",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "open_acc",
    "pub_rec",
    "purpose",
    "revol_bal",
    "revol_util",
    "sub_grade",
    "term",
    "total_acc",
    "loan_status"
]


def perform_neural_net_optimization(graph_map):
    train = pd.read_csv("data/lendingclub_train.csv", usecols=columns)
    test = pd.read_csv("data/lendingclub_test.csv", usecols=columns)

    train = train.sample(1000, random_state=7)

    training_features = train[columns[:-1]]
    training_classes = train[columns[-1:]].astype(np.bool)
    test_features = test[columns[:-1]]
    test_classes = test[columns[-1:]].astype(np.bool)

    graph_directory = 'neural_net_optimization_graphs'

    algorithms = ['random_hill_climb', 'simulated_annealing',
                  'genetic_alg']
    function_name = 'neural_net_fitness'
    if function_name not in graph_map:
        graph_map[function_name] = {}
    for algorithm in algorithms:
        # max attempts
        parameter = 'max_attempts'
        if parameter not in graph_map[function_name]:
            graph_map[function_name][parameter] = {}
        intervals = 20
        interval_size = 2
        training_scores = []
        for i in range(1, intervals + 1):
            max_attempts = i * interval_size
            model = mlrose.NeuralNetwork(hidden_nodes=[4],
                                         activation='relu',
                                         algorithm=algorithm,
                                         learning_rate=0.001,
                                         max_attempts=max_attempts,
                                         schedule=mlrose.ExpDecay(),
                                         restarts=6,
                                         early_stopping=True,
                                         random_state=7)
            model.fit(training_features, training_classes)
            predicted_training_classes = model.predict(training_features)
            predicted_test_classes = model.predict(test_features)
            training_score = accuracy_score(training_classes, predicted_training_classes)
            training_error = 1.0 - training_score
            test_score = accuracy_score(test_classes, predicted_test_classes)
            test_error = 1.0 - test_score
            training_scores.append([max_attempts, test_error, training_error])
        graph_map[function_name][parameter][algorithm] = training_scores
        plot_frame = pd.DataFrame(training_scores, columns=[parameter, 'Test Error',
                                                           'Training Error'])
        title = algorithm + '-' + parameter
        print(title)
        graph = plot_frame.plot(x=parameter, y=['Test Error', 'Training Error'],
                                title=title)
        graph.set_xlabel(parameter)
        graph.set_ylabel('Error')
        plt.savefig(graph_directory + '/' + title + '.png')
        # max iters
        parameter = 'max_iters'
        if parameter not in graph_map[function_name]:
            graph_map[function_name][parameter] = {}
        intervals = 20
        interval_size = 20
        training_scores = []
        for i in range(1, intervals + 1):
            max_iters = i * interval_size
            model = mlrose.NeuralNetwork(hidden_nodes=[4],
                                         activation='relu',
                                         algorithm=algorithm,
                                         learning_rate=0.001,
                                         schedule=mlrose.ExpDecay(),
                                         restarts=5,
                                         max_iters=max_iters,
                                         random_state=7)
            model.fit(training_features, training_classes)
            predicted_training_classes = model.predict(training_features)
            predicted_test_classes = model.predict(test_features)
            training_score = accuracy_score(training_classes, predicted_training_classes)
            training_error = 1.0 - training_score
            test_score = accuracy_score(test_classes, predicted_test_classes)
            test_error = 1.0 - test_score
            training_scores.append([max_iters, test_error, training_error])
        graph_map[function_name][parameter][algorithm] = training_scores
        plot_frame = pd.DataFrame(training_scores, columns=[parameter, 'Test Error',
                                                           'Training Error'])
        title = algorithm + '-' + parameter
        print(title)
        graph = plot_frame.plot(x=parameter, y=['Test Error', 'Training Error'], title=title)
        graph.set_xlabel(parameter)
        graph.set_ylabel('Error')
        plt.savefig(graph_directory + '/' + title + '.png')
        # time
        parameter = 'training_time'
        if parameter not in graph_map[function_name]:
            graph_map[function_name][parameter] = {}
        if 'testing_time' not in graph_map[function_name]:
            graph_map[function_name]['testing_time'] = {}
        intervals = 20
        training_scores = []
        testing_scores = []
        for i in range(0, intervals):
            start = timer()
            model = mlrose.NeuralNetwork(hidden_nodes=[4],
                                         activation='relu',
                                         algorithm=algorithm,
                                         learning_rate=0.001,
                                         schedule=mlrose.ExpDecay(),
                                         restarts=5,
                                         random_state=7)
            model.fit(training_features, training_classes)
            end = timer()
            time = end - start
            training_scores.append([i, time])
            start = timer()
            predicted_test_classes = model.predict(test_features)
            end = timer()
            time = end - start
            testing_scores .append([i, time])
        graph_map[function_name]['training_time'][algorithm] = training_scores
        graph_map[function_name]['testing_time'][algorithm] = testing_scores
        plot_frame = pd.DataFrame(training_scores, columns=[parameter, "Fitness"])
        title = algorithm + '-' + parameter
        print(title)
        graph = plot_frame.plot(x=parameter, y='Fitness', title=title)
        graph.set_xlabel(parameter)
        graph.set_ylabel("Fitness")
        plt.savefig(graph_directory + '/' + title + '.png')
        parameter = 'testing_time'
        plot_frame = pd.DataFrame(training_scores, columns=[parameter, "Fitness"])
        title = algorithm + '-' + parameter
        print(title)
        graph = plot_frame.plot(x=parameter, y='Fitness', title=title)
        graph.set_xlabel(parameter)
        graph.set_ylabel("Fitness")
        plt.savefig(graph_directory + '/' + title + '.png')
