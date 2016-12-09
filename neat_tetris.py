import os
import tetris
from neat import nn, population, statistics, parallel

# Network inputs and expected outputs.
xor_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
xor_outputs = [0, 1, 1, 0]

def continous_evaluation(tetris_session, net):
    if not tetris_session.gg:
        inputs = []

        for row_index, row in enumerate(tetris_session.matrix.matrix):
            for column_index, column in enumerate(row):
                if column == 2:
                    inputs.append(1)
                    tetris_session.matrix.matrix[row_index][column_index] = 1
                else:
                    inputs.append(0)
        for row in tetris_session.matrix.matrix:
            for column in row:
                inputs.append(column)

        output = net.serial_activate(inputs)
        if (output[0] > 0.5):  # left
            tetris_session.current_shape.move(-1, 0)
        if (output[1] > 0.5):  # right
            tetris_session.current_shape.move(1, 0)
        if (output[2] > 0.5):  # down
            tetris_session.current_shape.move(0, 1)
        if (output[3] > 0.5):  # rotate
            tetris_session.current_shape.rotate()
    else:
        tetris_session.root.destroy()


def eval_fitness(genome):
    net = nn.create_feed_forward_phenotype(genome)

    tetris_session = tetris.Game()
    tetris_session.start(continous_evaluation, net)

    if tetris_session.score > 30000:
        return 1
    else:
        created_shapes_bonus = min((tetris_session.created_shapes / 30) * 0.2, 0.2)
        return (tetris_session.score / (tetris_session.score + 30000)) * 0.8 + created_shapes_bonus


def run():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_tetris_config')
    pe = parallel.ParallelEvaluator(15, eval_fitness)
    pop = population.Population(config_path)
    pop.run(pe.evaluate, 1000000)

    print('Number of evaluations: {0}'.format(pop.total_evaluations))

    # Display the most fit genome.
    winner = pop.statistics.best_genome()
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