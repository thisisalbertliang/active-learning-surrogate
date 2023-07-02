#!/bin/bash

python -m al_surrogate.main \
    --target "gaussian-mixture" \
    --means "[[-10.0], [10.0]]" \
    --covars "[[[1.0]], [[1.0]]]" \
    --surrogate "gaussian-process" \
    --query-strategy "improved-greedy-sampling" \
    --batch-size 10 \
    --input-ranges "{'x': [-20.0, 20.0]}" \
    --num-active-learning-iterations 50
