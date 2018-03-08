#!/usr/bin/env bash
nohup python lagvae.py --lagrangian --gpu=0 --mi=-1.0 -z=5  &
nohup python lagvae.py --lagrangian --gpu=1 --mi=0.0 -z=5  &
nohup python lagvae.py --lagrangian --gpu=2 --mi=1.0 -z=5  &
nohup python lagvae.py --lagrangian --gpu=3 --mi=-1.0 -z=50  &
nohup python lagvae.py --lagrangian --gpu=0 --mi=0.0 -z=50  &
nohup python lagvae.py --lagrangian --gpu=1 --mi=1.0 -z=50  &
