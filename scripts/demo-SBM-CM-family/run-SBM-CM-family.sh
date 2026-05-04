#!/usr/bin/env bash
python3.11 scripts/demo-SBM-CM-family/SBM-CM-family.py CM data/MSA_array/MSA_CM.npy --TestTrain 0 --m 1 --rep 1 --N_av 1 --N_iter 400 --theta 0.3 --ParamInit zero --lambdJ 0 --lambdJ 0 --N_chains 70
