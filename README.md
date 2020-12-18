# sokoban_ai
Sokoban is a puzzle game where a single player pushes boxes to different storage locations in a warehouse map.  
Each map contains a list of movable boxes, immovable walls, a list of storage locations, and the player’s initial starting position. Players cannot move into immobile objects suchas walls, but can push boxes given that the box has an empty location to be pushed to. To complete the game, the player must push a box onto each storage location marked on the map.  
In this project, we try to build an AI agent which can solve Sokoban map.

## Running the code
We have implemented two algorithms.

For Q-Learning agent:  
`python3 sokoban.py train box <map_file>`

For Deep Q-Learning agent:  
`python3 sokoban.py train deep <map_file> --time 3600`  
**Note:** deep Q-learning agent may not converge. 

Runs with time limit:  
`python3 sokoban.py train box <map_file> --time 3600`


## More commands
Below, we provide a sample cut of the help output from the sokoban.py in the command line.
```
usage: sokoban.py [-h] [--quiet] [--verbose] [--episodes EPISODES]
                  [--iterations ITERATIONS] [--learning_rate LEARNING_RATE]
                  [--buffer_size BUFFER_SIZE]
                  [--minibatch_size MINIBATCH_SIZE] [--output OUTPUT]
                  [--save_figure] [--draw] [--sequence SEQUENCE] [--all]
                  [--pause PAUSE] [--time TIME]
                  [command [command ...]]

Solve a Sokoban game using artificial intelligence.

positional arguments:
  command

optional arguments:
  -h, --help            show this help message and exit
  --quiet, -q
  --verbose, -v
  --episodes EPISODES
  --iterations ITERATIONS
  --learning_rate LEARNING_RATE
  --buffer_size BUFFER_SIZE
  --minibatch_size MINIBATCH_SIZE
  --output OUTPUT, -o OUTPUT
  --save_figure, -s
  --draw, -d
  --sequence SEQUENCE
  --all
  --pause PAUSE
  --time TIME
```
