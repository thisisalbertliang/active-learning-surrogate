#!/bin/bash

python -m al_surrogate.main \
    --target "gaussian-mixture" \
    --means "[[10.0, 10.0], [-10.0, -10.0]]" \
    --covars "[[[2.0, 0.0], [0.0, 2.0]], [[1.0, 0.0], [0.0, 1.0]]]" \
    --surrogate "gaussian-process" \
    --query-strategy "uniform-sampling" \
    --batch-size 10 \
    --input-ranges "{'x': [-20.0, 20.0], 'y': [-20.0, 20.0]}"
