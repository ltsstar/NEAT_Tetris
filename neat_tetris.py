import os
import tetris
from neat import nn, population, statistics, parallel
import copy
import time
import pickle

def continous_evaluation(tetris_session, net):
    if not tetris_session.gg:
        inputs = []
        #deepcopy is far too slow, make it by our own
        matrix = []
        for row in tetris_session.matrix.matrix:
            matrix.append([])
            for column in row:
                matrix[-1].append(column)
        for row_index, row in enumerate(matrix):
            for column_index, column in enumerate(row):
                if column == 2:
                    inputs.append(1)
                    matrix[row_index][column_index] = 1
                else:
                    inputs.append(0)
        for row in matrix:
            for column in row:
                inputs.append(column)

        output = net.serial_activate(inputs)
        if (output[0] > 0.5):  # left
            #tetris_session.current_shape.move(-1, 0)
            tetris_session.matrix.left()
        if (output[1] > 0.5):  # right
            #tetris_session.current_shape.move(1, 0)
            tetris_session.matrix.right()
        if (output[2] > 0.5):  # down
            #tetris_session.current_shape.move(0, 1)
            tetris_session.matrix.fall()
        if (output[3] > 0.5):  # rotate
            #tetris_session.current_shape.rotate()
            tetris_session.matrix.rotate()
    else:
        #tetris_session.root.destroy()
        pass


def eval_fitness(genome):
    net = nn.create_feed_forward_phenotype(genome)

    score = 0
    for i in range(10): #do 10 games
        tetris_session = tetris.Game()
        tetris_session.start(continous_evaluation, net, False)
        score += tetris_session.score

    if score > 30000:
        return 1
    else:
        return (score / (score + 30000))

def eval_fitness_genomes(genomes):
    fitness = 0
    for g in genomes:
        g.fitness = eval_fitness(g)
        if g.fitness >= fitness:
            best_genome = g
    f = open("best_genome.pkl", "wb")
    pickle.dump(best_genome, f)
    f.close()

def run():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_tetris_config')
    pe = parallel.ParallelEvaluator(50, eval_fitness)
    pop = population.Population(config_path)
    pop.run(pe.evaluate, 1000000)
    #pop.run(eval_fitness_genomes, 100)

    print('Number of evaluations: {0}'.format(pop.total_evaluations))

    # Display the most fit genome.
    winner = pop.statistics.best_genome()
    f = open("best_genome.pkl", "wb")
    pickle.dump(winner, f)
    f.close()
    print('\nBest genome:\n{!s}'.format(winner))

    # Verify network output against training data.
    print('\nOutput:')
    winner_net = nn.create_feed_forward_phenotype(winner)
    for inputs, expected in zip(xor_inputs, xor_outputs):
        output = winner_net.serial_activate(inputs)
        print("expected {0:1.5f} got {1:1.5f}".format(expected, output[0]))

    '''
    # Visualize the winner network and plot/log statistics.
    visualize.plot_stats(pop.statistics)
    visualize.plot_species(pop.statistics)
    visualize.draw_net(winner, view=True, filename="xor2-all.gv")
    visualize.draw_net(winner, view=True, filename="xor2-enabled.gv", show_disabled=False)
    visualize.draw_net(winner, view=True, filename="xor2-enabled-pruned.gv", show_disabled=False, prune_unused=True)
    statistics.save_stats(pop.statistics)
    statistics.save_species_count(pop.statistics)
    statistics.save_species_fitness(pop.statistics)
    '''
if __name__ == '__main__':
    run()