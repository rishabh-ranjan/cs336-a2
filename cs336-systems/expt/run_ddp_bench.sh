#!/bin/bash

set -x

for lm_size in small medium large xl 2.7b
do
	# no ddp
	python -m ddp_bench --lm_size $lm_size --ddp 0

	# naive ddp
	torchrun --standalone --nnodes 1 --nproc_per_node 2 \
		-m ddp_bench --lm_size $lm_size --naive 1

	# ddp with flat(int), overlap individual (0), bucketing
	for bucket_size_mb in inf 0 5 10 50 100 500
	do
		torchrun --standalone --nnodes 1 --nproc_per_node 2 \
			-m ddp_bench --lm_size $lm_size --bucket_size_mb $bucket_size_mb
	done
done



