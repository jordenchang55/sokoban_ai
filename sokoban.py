import argparse
import csv
from pathlib import Path
from time import process_time

import matplotlib.pyplot as plt
import numpy as np

from environment import Environment

iteration_max = 1000  # deadlock by iteration
import random

deep_keyword = "deep"
box_keyword = "box"
q_keyword = "q"


def load(filename):
    filepath = Path(filename)
    # print(args.filename)
    if not filepath.exists():
        raise ValueError("Path does not exist.")
    if not filepath.is_file():
        raise ValueError("Path is not a valid file.")

    with open(filepath, 'r') as file:
        csv_input = csv.reader(file, delimiter=' ')

        for index, row in enumerate(csv_input):

            def unpack(points):

                return [tuple([int(points[index + 1]), int(points[index])]) for index in range(1, len(points) - 1, 2)]

            # print(index, row)

            if index == 0:
                # sizeH, sizeV

                xlim = int(row[0])
                ylim = int(row[1])
            if index == 1:
                # print(MapType.WALL.value)
                walls = unpack(row)
            elif index == 2:
                boxes = unpack(row)
            elif index == 3:
                storage = unpack(row)
            elif index == 4:
                player = np.array([int(row[1]), int(row[0])])

    return walls, boxes, storage, player, xlim, ylim


def create_env_agent(agent_name, filename):
    """
    Create environment and agent by given map file and agent name.
    """
    walls, boxes, storage, player, xlim, ylim = load(filename)
    print(f"Create env {args.command[2]}:({xlim},{ylim}) with {len(boxes)} boxes")
    if agent_name == "deep":
        from deepqagent import DeepQAgent
        from deepenvironment import DeepEnvironment
        environment = DeepEnvironment(filename = filename, walls=walls, boxes=boxes, storage=storage, player=player, xlim=xlim, ylim=ylim,
                                      pause=args.pause)
        agent = DeepQAgent(environment=environment, discount_factor=0.95, verbose=args.verbose)
    elif agent_name == "box":
        from boxagent import BoxAgent
        from stateenvironment import StateEnvironment
        environment = StateEnvironment(filename = filename,walls=walls, boxes=boxes, storage=storage, player=player, xlim=xlim, ylim=ylim,
                                       pause=args.pause)
        agent = BoxAgent(environment=environment, discount_factor=0.95, verbose=args.verbose)
    elif agent_name == "q":
        from agent import QAgent
        from stateenvironment import StateEnvironment
        environment = StateEnvironment(filename = filename, walls=walls, boxes=boxes, storage=storage, player=player, xlim=xlim, ylim=ylim,
                                       pause=args.pause)
        agent = QAgent(environment=environment, discount_factor=0.95, verbose=args.verbose)
    else:
        raise ValueError("Unknown agent name: %s" % agent_name)
    return environment, agent


def train_all():
    """
    randomly train on the different maps for the given amount of episodes...
    """
    input_path = Path(args.command[2])
    if input_path.exists() and not input_path.is_dir():
        raise ValueError("Should be directory.")
    file_list = list(input_path.glob('*'))

    max_episodes = abs(args.episodes)
    max_iterations = abs(args.iterations)

    if args.command[1] != "deep":
        raise NotImplementedError
    from deepenvironment import DeepEnvironment

    print(args.command[3])
    environment, agent = create_env_agent(args.command[1], file_list[0])
    if len(args.command) >= 4:
        pretrain_path = Path("sokoban_state.pth")
        if pretrain_path.exists() and pretrain_path.is_file():
            agent.load("sokoban_state.pth")
        elif pretrain_path.exists() and not pretrain_path.is_file():
            raise ValueError("Invalid pytorch file.")
    else:
        pretrain_path = Path(args.command[3])
        if pretrain_path.exists() and pretrain_path.is_file():
            agent.load(args.command[3])
        elif pretrain_path.exists() and not pretrain_path.is_file():
            raise ValueError("Invalid file input.")

    epochs = 0
    max_epochs = 5000

    while epochs < max_epochs:
        file = random.choice(file_list)
        walls, boxes, storage, player, xlim, ylim = load(file)

        while epochs < 5 and (xlim >= 9 or ylim >= 9):
            file = random.choice(file_list)
            walls, boxes, storage, player, xlim, ylim = load(file)

        if args.verbose:
            print(f"epoch {epochs}:{file} of size {xlim}, {ylim}.")

        environment = DeepEnvironment(filename=file, walls=walls, boxes=boxes, storage=storage, player=player, xlim=xlim, ylim=ylim)
        agent.load_environment(environment)

        while True:

            goal, iterations, actions = agent.episode(draw=args.draw, evaluate=False, max_iterations=max_iterations)

            if agent.num_episodes % 100 == 0:

                goal, iterations, actions = agent.episode(draw=args.draw, evaluate=True, max_iterations=200)
                if goal:
                    break

        if len(args.command) >= 4:
            agent.save(args.command[3])
        else:
            agent.save("sokoban_state.pth")

        epochs += 1

    if len(args.command) >= 4:
        agent.save(args.command[3])
    else:
        agent.save("sokoban_state.pth")

    with open('losses.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for loss in agent.losses:
            writer.writerow(loss)


def train():
    """
    sokoban.py train <agent> <input file>
    """
    if len(args.command) < 3:
        raise Exception("Expected agent and filepath input.")
    if args.all:
        train_all()
        return None

    # import matplotlib.patches as patches

    max_episodes = abs(args.episodes)
    max_iterations = abs(args.iterations)

    start_time = process_time()

    walls, boxes, storage, player, xlim, ylim = load(args.command[2])

    environment, agent = create_env_agent(args.command[1], args.command[2])
    if args.command[1] == "deep":
        if len(args.command) < 4:
            pretrain_path = Path("sokoban_state.pth")
            if pretrain_path.exists() and pretrain_path.is_file():
                agent.load("sokoban_state.pth")
            elif pretrain_path.exists() and not pretrain_path.is_file():
                raise ValueError("Invalid pytorch file.")
        else:
            pretrain_path = Path(args.command[3])
            if pretrain_path.exists() and pretrain_path.is_file():
                agent.load(args.command[3])
            elif pretrain_path.exists() and not pretrain_path.is_file():
                raise ValueError("Invalid file input.")

    goal_evaluated = False
    while agent.num_episodes < max_episodes:
        # if num_episodes % 500 == 0 and num_episodes > 0:
        #   iterative_threshold = iterative_threshold*2

        goal, iterations, _ = agent.episode(draw=args.draw, evaluate=False, max_iterations=max_iterations)

        if args.command[1] == "box":
            if agent.num_episodes > 0 and agent.num_episodes % 100 == 0:
                goal_evaluated, iterations, _ = agent.episode(draw=args.verbose, evaluate=True, max_iterations=200)
        elif args.command[1] == "q":
            if agent.num_episodes > 0 and agent.num_episodes % 10 == 0:
                goal_evaluated, iterations, _ = agent.episode(draw=args.draw, evaluate=True, max_iterations=200)
        elif args.command[1] == "deep":
            if agent.num_episodes > 0 and agent.num_episodes % 50 == 0:
                goal_evaluated, iterations, _ = agent.episode(draw=args.draw, evaluate=True, max_iterations=200)
        if goal_evaluated:
            break
    # num_episodes += 1

    if args.command[1] == "deep":
        if len(args.command) == 3:
            agent.save(args.command[2])
        else:
            agent.save("sokoban_state.pth")

    goal, iterations, actions = agent.episode(draw=True, evaluate=True, max_iterations=200)

    print("-" * 30)
    print("Simulation ended.")
    print(f"episodes         :{agent.num_episodes}")
    print(f"map solved       :{goal}")
    print(f"iterations       :{iterations}")
    print(f"deadlock hit rate:{environment.cache_hit / (environment.cache_miss + environment.cache_hit):.3f}")
    print(f"time taken       :{process_time() - start_time:.3f}")
    print("-" * 30)
    plt.show(block=True)


def evaluate():
    """
    sokoban.py evaluate <agent> <input>
    """
    max_episodes = abs(args.episodes)

    if len(args.command) < 3:
        raise Exception("Expected agent and filepath input.")

    environment, agent = create_env_agent(args.command[0], args.command[1])

    pretrain_path = Path("sokoban_state.pth")
    if pretrain_path.exists() and pretrain_path.is_file():
        agent.load("sokoban_state.pth")

    agent.episode(draw=False, evaluate=True)


def test():
    import unittest
    import tests.stateenvironmenttest
    import tests.deepenvironmenttest
    import tests.deepqagenttest

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromModule(tests.stateenvironmenttest))
    suite.addTests(loader.loadTestsFromModule(tests.deepqagenttest))
    suite.addTests(loader.loadTestsFromModule(tests.deepenvironmenttest))

    if args.verbose:
        verbose = 2
    elif args.quiet:
        verbose = 0
    else:
        verbose = 1
    runner = unittest.TextTestRunner(verbosity=verbose)
    result = runner.run(suite)


def draw():
    from stateenvironment import StateEnvironment

    if len(args.command) < 2:
        raise Exception("Expected a filepath argument.")

    def draw_file(filename):
        walls, boxes, storage, player, xlim, ylim = load(filename)
        environment = StateEnvironment(filename = filename, walls=walls, boxes=boxes, storage=storage, player=player, xlim=xlim, ylim=ylim)

        environment.draw(environment.state)
        # for action in [
        #   Environment.LEFT,
        #   Environment.DOWN,
        #   Environment.LEFT,
        #   Environment.LEFT,
        #   Environment.RIGHT,
        #   Environment.DOWN
        # ]:
        #   environment.state = environment.next_state(environment.state, action)
        #   environment.draw(environment.state)
        plt.show()
        plt.pause(5)

    if args.all:
        input_path = Path(args.command[1])
        if input_path.exists() and not input_path.is_dir():
            raise ValueError("Should be directory.")
        file_list = list(input_path.glob('*'))
        for f in file_list:
            draw_file(f)
    else:
        draw_file(args.command[1])


# if args.sequence:


def time():

    number_of_runs = 20

    if len(args.command) < 4:
        raise Exception("Expected 'sokoban.py time <agent> <input file> <output file>' format.")


    times = []
    for i in range(number_of_runs):
        print(i)
        converge_flag = False

        environment, agent = create_env_agent(agent_name = args.command[1], filename = args.command[2])


        start_time = process_time()
        while not converge_flag:
            goal, iterations, action_sequence = agent.episode(draw = False, evaluate=False, max_iterations=args.iterations)

            if agent.num_episodes % 20 == 0:
                converge_flag, _, _ = agent.episode(draw = False, evaluate=True, max_iterations=200)

        times.append(process_time()-start_time)
    data = zip([args.command[2]]*number_of_runs, [environment.xlim]*number_of_runs, [environment.ylim]*number_of_runs, [len(environment.state.boxes)]*number_of_runs, times)

    with open(args.command[3], 'a') as file:
        writer = csv.writer(file, delimiter=',')
        for datum in data:
            writer.writerow(datum)


def plot():
    data = []

    if len(args.command) < 2:
        raise Exception("Expected 'plot <csv file>' format.")

    with open(args.command[1], 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            data.append([str(row[0]), eval(row[1]), eval(row[2]), eval(row[3]), eval(row[4])])

    files, xlims, ylims, boxes, times = zip(*data)

    import re
    from pylab import plot, show, savefig, xlim, figure, \
                ylim, legend, boxplot, setp, axes


    parsed_files = []
    for file in files:
        match = re.search('([A-Za-z_]+([0-9]+[a-z]*)\.txt)', file)
        if match:
            parsed_files.append(match.group(2))

    timing_data = {}
    features = {}
    box_plots = []


    fig = figure()
    ax = axes()
    fig.suptitle('Benchmarks times across 20 samples for box agent')
    for file, x, y, num_box, t in zip(parsed_files, xlims, ylims, boxes, times):
        if file not in timing_data:
            timing_data[file] = []
        if file not in features:
            features[file] = (x, y, num_box)
        timing_data[file].append(t)



    ax.set_yscale('log')    
    for index, key in enumerate(timing_data):
        print(f"{key:4s}{features[key]}:{np.mean(timing_data[key]):.2f}+{np.std(timing_data[key]):.2f}")
        box_plots.append(boxplot(timing_data[key], positions=[(index)+1], widths=0.6))
        #setBoxColors(box_plots[-1])
    keys = [key for key in timing_data]

    timemax = np.array(times).max()
    ax.set_xticklabels(keys)
    ax.set_xticks(np.arange(1, len(keys)+1, 1))

    xlim(0,len(keys)+1)
    ylim(1,timemax*1.1)

    
    #plt.scatter(parsed_files, times, s=6)
    #show()
    filename = Path(args.command[1])
    savefig(f"{filename.stem}.png")
    #print(data)


def main():
    if args.command[0] == "train":
        train()
    elif args.command[0] == "test":
        test()
    elif args.command[0] == "draw":
        draw()
    elif args.command[0] == "evaluate":
        evaluate()
    elif args.command[0] == "time":
        time()
    elif args.command[0] == "plot":
        plot()
    else:
        print("Unrecognized command. Please use sokoban.py --help for help on usage.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Solve a Sokoban game using artificial intelligence.")
    parser.add_argument('--quiet', '-q', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--episodes', action='store', type=int, default=500)
    parser.add_argument('--iterations', action='store', type=int, default=5000)
    parser.add_argument('--learning_rate', action='store', type=float, default=1e-5)
    parser.add_argument('--buffer_size', action='store', type=int, default=5000000)
    parser.add_argument('--minibatch_size', action='store', type=int, default=128)

    parser.add_argument('--output', '-o', type=str)
    parser.add_argument('--save_figure', '-s', action='store_true')
    parser.add_argument('--draw', '-d', action='store_true')
    parser.add_argument('--sequence', type=str)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--pause', type=float, default=0.05)
    parser.add_argument('--time', type=int)
    parser.add_argument('command', nargs='*')

    args = parser.parse_args()
    main()
