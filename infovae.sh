#!/usr/bin/env bash
nohup python vae_lagrangian.py --gpu=0 --mi=-1.0 -l1=2.0 -l2=10.0 -e1=0.0 -e2=0.0 &
nohup python vae_lagrangian.py --gpu=1 --mi=0.0 -l1=2.0 -l2=10.0 -e1=0.0 -e2=0.0 &
nohup python vae_lagrangian.py --gpu=2 --mi=1.0 -l1=2.0 -l2=10.0 -e1=0.0 -e2=0.0 &
