SHELL:=/bin/bash

all:
	@echo "choose explicit target = type 'make ' and press TAB"

S=scripts
I=data
O=out


# ===== MAIN STUFF 

INPUT=specify_INPUT_on_command_line
FLAGS=-0
cluster:
	@echo "--- $@" 1>&2
	python3 $S/clustering_basics.py --file $(INPUT)_vectors.csv $(FLAGS) > $(INPUT)_results.csv

cluster_all:
	@echo "--- $@" 1>&2
	make cluster INPUT=t1 FLAGS=-n
	make cluster INPUT=t2 FLAGS=-n
	make cluster INPUT=t3 FLAGS=-n
	make cluster INPUT=nu FLAGS=-0
	make cluster INPUT=ik FLAGS=-0

