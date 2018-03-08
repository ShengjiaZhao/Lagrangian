#!/usr/bin/env bash
nohup python lagvae.py --gpu=0 --mi=0.0 -l1=1.0 -l2=0.0 -e1=0.0 -e2=0.0 -t=mlp -z=50 &
nohup python lagvae.py --gpu=1 --mi=0.0 -l1=1.0 -l2=0.0 -e1=0.0 -e2=0.0 -t=cnn -z=50 &
nohup python lagvae.py --gpu=0 --mi=0.0 -l1=1.0 -l2=0.0 -e1=0.0 -e2=0.0 -t=cnns -z=50 &