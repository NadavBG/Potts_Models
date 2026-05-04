#!/usr/bin/env bash

FULL_CM_ALG="../data/MSA_array/MSA_CM.npy"

# first, we build the masks based on the full alignment
python build_mask.py --alg $FULL_CM_ALG --theta 0.7 --lbda 0.03 --strategies "fij" "cij" "sca" \
                     --ext ".npy" --label "CM" --path "./prune_output" --percent 98

# next, we train the sBM models
python ../scripts/demo-SBM-CM-family/SBM-CM-family.py SCAPruned_CM $FULL_CM_ALG --TestTrain 0 \
       --m 20 --rep 1 --N_av 1 --N_iter 400 --theta 0.3 --ParamInit zero \
       --lambdJ 0.01 --lambdh 0.01 --N_chains 100 \
       --prune "./prune_output/98.00_SCA_CM_SeqW_0.7.npy" \
       --results_path "./example_output/"
