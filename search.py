import argparse

from structure import Node, Environment
from queue import PriorityQueue

FOUND = 1
NOT_FOUND = 0
LIMIT_REACHED = -1


def h(n: Node):
    # TODO: implement this
    return 0


def g(n: Node):
    return n.path_cost


def f(n):
    return h(n) + g(n)


def expand(env: Environment, node: Node) -> [Node]:
    """
    Expands given node to get its children on search tree.
    """
    for action in env.get_actions(node.state):
        next_state = env.move(node.state, action)
        cost = node.path_cost + env.calc_action_cost(node.state, action, next_state)
        yield Node(next_state, action, node, cost)


def best_first_search(env: Environment, func, limit: int = None):
    Node.get_value = func
    queue = PriorityQueue()
    visited = {}
    node = Node(env.state)
    queue.put(node)
    level = 0
    while not queue.empty():
        node = queue.get()
        env.state = node.state
        if env.is_goal(node):
            return FOUND, node
        for child in expand(env, node):
            if child not in visited or child.path_cost < visited[hash(node)].path_cost:
                visited[hash(node)] = node
                queue.put(node)
        level += 1
        if limit and level >= limit:
            return LIMIT_REACHED

    return NOT_FOUND


def ida_star(env: Environment, upper_bound: int = 100):
    for limit in range(1, upper_bound):
        res, node = best_first_search(env, f, limit)
        if res in (FOUND, NOT_FOUND):
            return res, node
    return LIMIT_REACHED


if __name__ == '__main__':
    import csv

    parser = argparse.ArgumentParser(description="Solve a Sokoban game using artificial intelligence.")
    parser.add_argument('filename')
    parser.add_argument('--save_figure', '-s', action='store_true')
    args = parser.parse_args()

    with open(args.filename, 'r') as file:
        csv_input = csv.reader(file, delimiter=' ')
        env = Environment(csv_input)
        best_first_search(env, g)
        env.draw(args.save_figure)
