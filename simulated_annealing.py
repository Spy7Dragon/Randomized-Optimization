import mlrose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer

graph_directory = "simulated_annealing_graphs"
algorithm = "simulated_annealing"
default_schedule = mlrose.ExpDecay()
best_max_iters = np.inf
best_max_attempts = 20


def perform_simulated_annealing_analysis(problems, graph_map):
    for fitness_func in problems:
        function_name = fitness_func.__name__
        fitness = mlrose.CustomFitness(fitness_func)
        problem = mlrose.DiscreteOpt(length=16, fitness_fn=fitness, maximize=True)
        # max_attempt
        parameter = 'default_max_attempt'
        if parameter not in graph_map[function_name]:
            graph_map[function_name][parameter] = {}
        intervals = 20
        interval_size = 1
        section_scores = []
        for i in range(1, intervals + 1):
            max_attempts = i * interval_size
            best_state, best_fitness = mlrose.simulated_annealing(problem,
                                                                  schedule=default_schedule,
                                                                  max_attempts=max_attempts,
                                                                  random_state=7)
            print(best_state, best_fitness)
            section_scores.append([max_attempts, best_fitness])
        graph_map[function_name][parameter][algorithm] = section_scores
        plot_frame = pd.DataFrame(section_scores, columns=[parameter, "Fitness"])
        title = function_name + '-' + parameter
        graph = plot_frame.plot(x=parameter, y='Fitness', title=title)
        graph.set_xlabel(parameter)
        graph.set_ylabel("Fitness")
        plt.savefig(graph_directory + '/' + title + '.png')
        parameter = 'best_max_attempt'
        if parameter not in graph_map[function_name]:
            graph_map[function_name][parameter] = {}
        intervals = 20
        interval_size = 1
        section_scores = []
        for i in range(1, intervals + 1):
            max_attempts = i * interval_size
            best_state, best_fitness = mlrose.simulated_annealing(problem,
                                                                  schedule=default_schedule,
                                                                  max_attempts=max_attempts,
                                                                  max_iters=best_max_iters,
                                                                  random_state=7)
            print(best_state, best_fitness)
            section_scores.append([max_attempts, best_fitness])
        graph_map[function_name][parameter][algorithm] = section_scores
        plot_frame = pd.DataFrame(section_scores, columns=[parameter, "Fitness"])
        title = function_name + '-' + parameter
        graph = plot_frame.plot(x=parameter, y='Fitness', title=title)
        graph.set_xlabel(parameter)
        graph.set_ylabel("Fitness")
        plt.savefig(graph_directory + '/' + title + '.png')
        # max_iters
        parameter = 'default_max_iters'
        if parameter not in graph_map[function_name]:
            graph_map[function_name][parameter] = {}
        intervals = 20
        interval_size = 1
        section_scores = []
        for i in range(1, intervals + 1):
            max_iters = i * interval_size
            best_state, best_fitness = mlrose.simulated_annealing(problem,
                                                                  schedule=default_schedule,
                                                                  max_iters=max_iters,
                                                                  random_state=7)
            print(best_state, best_fitness)
            section_scores.append([max_iters, best_fitness])
        graph_map[function_name][parameter][algorithm] = section_scores
        plot_frame = pd.DataFrame(section_scores, columns=[parameter, "Fitness"])
        title = function_name + '-' + parameter
        graph = plot_frame.plot(x=parameter, y='Fitness', title=title)
        graph.set_xlabel(parameter)
        graph.set_ylabel("Fitness")
        plt.savefig(graph_directory + '/' + title + '.png')
        parameter = 'best_max_iters'
        if parameter not in graph_map[function_name]:
            graph_map[function_name][parameter] = {}
        intervals = 20
        interval_size = 1
        section_scores = []
        for i in range(1, intervals + 1):
            max_iters = i * interval_size
            best_state, best_fitness = mlrose.simulated_annealing(problem,
                                                                  schedule=default_schedule,
                                                                  max_iters=max_iters,
                                                                  max_attempts=best_max_attempts,
                                                                  random_state=7)
            print(best_state, best_fitness)
            section_scores.append([max_iters, best_fitness])
        graph_map[function_name][parameter][algorithm] = section_scores
        plot_frame = pd.DataFrame(section_scores, columns=[parameter, "Fitness"])
        title = function_name + '-' + parameter
        graph = plot_frame.plot(x=parameter, y='Fitness', title=title)
        graph.set_xlabel(parameter)
        graph.set_ylabel("Fitness")
        plt.savefig(graph_directory + '/' + title + '.png')
        # time
        parameter = 'time'
        if parameter not in graph_map[function_name]:
            graph_map[function_name][parameter] = {}
        intervals = 20
        section_scores = []
        for i in range(0, intervals):
            start = timer()
            best_state, best_fitness = mlrose.simulated_annealing(problem,
                                                                  schedule=default_schedule,
                                                                  max_iters=best_max_iters,
                                                                  max_attempts=best_max_attempts,
                                                                  random_state=7)
            end = timer()
            time = end - start
            section_scores.append([i, time])
        graph_map[function_name][parameter][algorithm] = section_scores
        plot_frame = pd.DataFrame(section_scores, columns=[parameter, "Fitness"])
        title = function_name + '-' + parameter
        graph = plot_frame.plot(x=parameter, y='Fitness', title=title)
        graph.set_xlabel(parameter)
        graph.set_ylabel("Fitness")
        plt.savefig(graph_directory + '/' + title + '.png')
