#!/bin/bash

for i in {0..61}
do
	string=$(printf "inputs/sokoban%02d.txt" $i)
	echo $string
	python3 sokoban.py train $string -e 1000 -i 6000
done
