#!/bin/bash
source ~/venv/pytorch/bin/activate
for i in {0..61}
do
	string=$(printf "inputs/sokoban%02d.txt" $i)
	echo $string
	python3 sokoban.py train $string --episodes 1000 --iterations 8000
done
