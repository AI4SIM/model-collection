#!/bin/bash

singularity exec \
        -H $WORK_DIR/run \
        --bind $WORK_DIR:$WORK_DIR \
        $OCI_MESONH \
        $MESONH_EXE