import argparse
import csv
from pathlib import Path

from structure import Environment


def main():
    filepath = Path(args.filename)
    # print(args.filename)
    if not filepath.exists():
        raise ValueError("Path does not exist.")
    if not filepath.is_file():
        raise ValueError("Path is not a valid file.")

    with open(filepath, 'r') as file:
        csv_input = csv.reader(file, delimiter=' ')
        env = Environment(csv_input)
        env.draw(args.save_figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Solve a Sokoban game using artificial intelligence.")
    parser.add_argument('filename')
    parser.add_argument('--save_figure', '-s', action='store_true')
    args = parser.parse_args()
    main()
