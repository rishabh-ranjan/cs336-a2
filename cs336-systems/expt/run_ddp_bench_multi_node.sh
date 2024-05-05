#!/bin/bash

set -x

for lm_size in small medium large xl 2.7b
do
	# naive ddp
	torchrun --nnodes 2 --nproc_per_node 1 --rdzv_id 0 --rdzv_backend c10d --rdzv_endpoint ampere1.stanford.edu:29401 \
		-m ddp_bench --lm_size $lm_size --naive 1

	# ddp with flat(int), overlap individual (0), bucketing
	for bucket_size_mb in inf 0 5 10 50 100 500
	do
		torchrun --nnodes 2 --nproc_per_node 1 --rdzv_id 0 --rdzv_backend c10d --rdzv_endpoint ampere1.stanford.edu:29401 \
			-m ddp_bench --lm_size $lm_size --bucket_size_mb $bucket_size_mb
	done
done
