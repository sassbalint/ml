SHELL:=/bin/bash

all:
	@echo "choose explicit target = type 'make ' and press TAB"

S=scripts
I=data
O=out


# ===== MAIN STUFF 

INPUT=specify_INPUT_on_command_line
clustering:
	@echo "--- $@" 1>&2
	python3 clustering_basics.py $(INPUT)_vectors.csv > $(INPUT)_results.csv

