#!/bin/zsh
make
python ../indexer.py $1
./indexer -p $1.pde -t $1.jos
