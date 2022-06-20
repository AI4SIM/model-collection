#!/bin/bash

singularity exec \
        -H $WORK_DIR/run \
        --bind $WORK_DIR:$WORK_DIR \
        $OCI_CROCO_XIOS \
        $XIOS_EXE