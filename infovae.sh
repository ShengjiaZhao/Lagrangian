#!/usr/bin/env bash
nohup python lagvae.py --gpu=0 --mi=-1.0 -l1=1.0 -l2=10.0 -e1=0.0 -e2=0.0 &
nohup python lagvae.py --gpu=1 --mi=0.0 -l1=1.0 -l2=10.0 -e1=0.0 -e2=0.0 &
nohup python lagvae.py --gpu=2 --mi=1.0 -l1=1.0 -l2=10.0 -e1=0.0 -e2=0.0 &

nohup python lagvae.py --gpu=3 --mi=-1.0 -l1=1.0 -l2=10.0 -e1=0.0 -e2=0.0 -z=50 &
nohup python lagvae.py --gpu=1 --mi=0.0 -l1=1.0 -l2=10.0 -e1=0.0 -e2=0.0 -z=50 &
nohup python lagvae.py --gpu=2 --mi=1.0 -l1=1.0 -l2=10.0 -e1=0.0 -e2=0.0 -z=50 &

nohup python lagvae.py --gpu=0 --mi=-1.0 -l1=1.0 -l2=10.0 -e1=0.0 -e2=0.0 -t=mlp &
nohup python lagvae.py --gpu=3 --mi=0.0 -l1=1.0 -l2=10.0 -e1=0.0 -e2=0.0 -t=mlp &
nohup python lagvae.py --gpu=3 --mi=1.0 -l1=1.0 -l2=10.0 -e1=0.0 -e2=0.0 -t=mlp &

nohup python lagvae.py --gpu=0 --mi=-1.0 -l1=1.0 -l2=10.0 -e1=0.0 -e2=0.0 -z=50 -t=mlp&
nohup python lagvae.py --gpu=1 --mi=0.0 -l1=1.0 -l2=10.0 -e1=0.0 -e2=0.0 -z=50 -t=mlp&
nohup python lagvae.py --gpu=2 --mi=1.0 -l1=1.0 -l2=10.0 -e1=0.0 -e2=0.0 -z=50 -t=mlp&