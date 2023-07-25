#!/bin/bash

DATA_FILE=queries_uniform.jsonl
DRAWING_FILE=queries_uniform.pdf

if [ "$1" != "--only-plot" ]; then
	mkdir -p results
	rm -f results/$DATA_FILE

	for n in 500 1000 1500 2000
	do
		q=$((20*n))
		echo "Benchmark queries with $n vertices"...
		for _ in {1..5}
		do
			s=$RANDOM
			echo "  seed=$s"
			./stt-benchmarks/target/release/bench_queries -s $s -n $n -q $q --json link-cut greedy-splay stable-greedy-splay two-pass-splay stable-two-pass-splay local-two-pass-splay local-stable-two-pass-splay move-to-root stable-move-to-root one-cut petgraph-dynamic >> results/$DATA_FILE || exit
		done
	done
fi

python3 show_benchmarks/visualize.py --input-file results/$DATA_FILE --profile queries-uniform --output-file results/$DRAWING_FILE
