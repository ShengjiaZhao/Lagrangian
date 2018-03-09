#!/usr/bin/env bash
lagvae() {
    nohup python mmd_vae.py --gpu=$1 --mi=-1.0 --lagrangian -e1=$2 -e2=$3 &
    nohup python mmd_vae.py --gpu=$1 --mi=0.0 --lagrangian -e1=$2 -e2=$3 &
    nohup python mmd_vae.py --gpu=$1 --mi=1.0 --lagrangian -e1=$2 -e2=$3 &
}

# atlas2
# lagvae 0 88.0 0.0007
# lagvae 1 90.0 0.0007

# atlas1
# lagvae 0 92.0 0.0007
# lagvae 1 88.0 0.0009

# atlas3
# lagvae 1 90.0 0.0009

# naivetoad
# lagvae 0 92.0 0.0009
# lagvae 1 88.0 0.0011
# lagvae 2 90.0 0.0011
# lagvae 3 92.0 0.0011

