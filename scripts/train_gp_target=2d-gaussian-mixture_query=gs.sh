#!/bin/bash

script_name=$(basename $0)
python -m al_surrogate.main \
    --target "gaussian-mixture" \
    --means "[[10.0, 10.0], [-10.0, -10.0]]" \
    --covars "[[[2.0, 0.0], [0.0, 2.0]], [[1.0, 0.0], [0.0, 1.0]]]" \
    --surrogate "gaussian-process" \
    --query-strategy "greedy-sampling" \
    --batch-size 10 \
    --input-ranges "{'x': [-20.0, 20.0], 'y': [-20.0, 20.0]}" \
    --num-active-learning-iterations 50 \
    --experiment-name $script_name
