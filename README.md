# sokoban_ai

In this assignment, you are asked to submit a software design for the AI (smarts) of the game of Sokoban (https://en.wikipedia.org/wiki/Sokoban).

## Running the code
Below, we provide a sample cut of the help output from the sokoban.py in the command line.
```
usage: sokoban.py [-h] [--quiet] [--verbose] [--episodes EPISODES] [--save_figure] [--draw] [--sequence SEQUENCE] [command [command ...]]

Solve a Sokoban game using artificial intelligence.

positional arguments:
  command

optional arguments:
  -h, --help            show this help message and exit
  --quiet, -q
  --verbose, -v
  --episodes EPISODES, -e EPISODES
  --save_figure, -s
  --draw, -d
  --sequence SEQUENCE
```



Sample commands to use:

`python3 sokoban.py draw <filename>`

Draws a given input. Action sequence replay functionality coming later.

`python3 sokoban.py test --verbose` 

Runs all unit tests for the sokoban source code. 

`python3 sokoban.py run <filename>`

Runs the agent on the given sokoban input map.
