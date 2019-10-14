from randomized_hill_climbing import *
from simulated_annealing import *
from genetic_algorithm import *
from mimic import *
from neural_net_optimization import *

answer = [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]


def genetic_algorithm_fitness(input):
    # 4 Queens
    queen_count = 0
    score = 0
    row_counts = [0, 0, 0, 0]
    col_counts = [0, 0, 0, 0]
    for row in range(0, 4):
        for col in range(0, 4):
            index = row * 4 + col
            if input[index] == 1:
                row_counts[row] += 1
                col_counts[col] += 1
                queen_count += 1

    if queen_count == 4:
        for i in range(0, 4):
            if row_counts[i] > 1:
                score -= (row_counts[i] - 1)
            if col_counts[i] > 1:
                score -= (col_counts[i] - 1)
    else:
        score = -100
    return score


def simulated_annealing_fitness(input):
    score = 0
    length = len(answer)
    # Easy problem with deterred neighbors
    for i in range(length):
        if input[i] == answer[i]:
            score += 1
    # Deter good neighbors
    if score != length:
        for i in range(length):
            if input[i] == 1 and input[i] == answer[i]:
                score -= 1
    return score


def mimic_fitness(input):
    # something complex
    score = 0
    length = len(answer)
    for i in range(length):
        if answer[i] == input[i]:
            score += (i + 1)
    return score


def create_comparison(graph_map):
    for function_name in graph_map:
        for parameter in graph_map[function_name]:
            algorithms = graph_map[function_name][parameter].keys()
            title = function_name + '-' + parameter
            curves = []
            index = []
            need_index = True
            if len(algorithms) > 1:
                for algorithm in algorithms:
                    section_scores = graph_map[function_name][parameter][algorithm]
                    curve = []
                    for section in section_scores:
                        if need_index:
                            index.append(section[0])
                        curve.append(section[1])
                    curves.append(curve)
                    need_index = False
                plot_frame = pd.DataFrame(curves, index=list(algorithms)).transpose()
                plot_frame.index = index
                graph = plot_frame.plot(y=algorithms, title=title)
                graph.set_xlabel(parameter)
                graph.set_ylabel('Fitness')
                plt.savefig('compare_graphs/' + title + '.png')


if __name__ == '__main__':
    graph_map = {}
    # problems = [genetic_algorithm_fitness, simulated_annealing_fitness, mimic_fitness]
    # perform_randomized_hill_climbing_analysis(problems, graph_map)
    # perform_simulated_annealing_analysis(problems, graph_map)
    # perform_genetic_algorithm_analysis(problems, graph_map)
    # perform_mimic_analysis(problems, graph_map)
    perform_neural_net_optimization(graph_map)
    create_comparison(graph_map)