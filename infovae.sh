#!/usr/bin/env bash
infovae() {
    nohup python mmd_vae.py --gpu=$1 --mi=-1.0 -l1=$2 -l2=$3 &
    nohup python mmd_vae.py --gpu=$1 --mi=0.0 -l1=$2 -l2=$3 &
    nohup python mmd_vae.py --gpu=$1 --mi=1.0 -l1=$2 -l2=$3 &
}

# atlas4
# infovae 0 1.0 100.0
# infovae 0 1.0 1000.0

# atlas5
infovae 0 1.0 10000.0
infovae 0 5.0 100.0