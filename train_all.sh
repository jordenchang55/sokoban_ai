#!/bin/bash
cd ~/git/sokoban/
source ~/venv/pytorch/bin/activate
for i in {0..1000}
do
        rand=$RANDOM
        let "rand %= 62"
	string=$(printf "inputs/sokoban%02d.txt" $rand)
	echo $string
	python3 sokoban.py train $string --episodes 100 --iterations 8000
done
