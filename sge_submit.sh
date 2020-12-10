#!/bin/bash
qsub -q opengpu.q -M tajk@ics.uci.edu -m beas -o train_full.$JOB_ID.out -e train_full.$JOB_ID.err ~/git/sokoban/train_all.sh
