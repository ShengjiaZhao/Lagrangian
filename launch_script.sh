#!/usr/bin/env bash
#nohup python vae_lagrangian.py --lagrangian --gpu=0 --mi=-1.0 -z=5  &
#nohup python vae_lagrangian.py --lagrangian --gpu=1 --mi=0.0 -z=5  &
#nohup python vae_lagrangian.py --lagrangian --gpu=2 --mi=1.0 -z=5  &
#nohup python vae_lagrangian.py --lagrangian --gpu=3 --mi=-1.0 -z=50  &
#nohup python vae_lagrangian.py --lagrangian --gpu=0 --mi=0.0 -z=50  &
#nohup python vae_lagrangian.py --lagrangian --gpu=1 --mi=1.0 -z=50  &
#nohup python vae_lagrangian.py --lagrangian --gpu=2 --mi=-1.0 -z=5 -t=mlp &
#nohup python vae_lagrangian.py --lagrangian --gpu=3 --mi=0.0 -z=5 -t=mlp &
#nohup python vae_lagrangian.py --lagrangian --gpu=0 --mi=1.0 -z=5 -t=mlp &
#nohup python vae_lagrangian.py --lagrangian --gpu=1 --mi=-1.0 -z=50 -t=mlp &
#nohup python vae_lagrangian.py --lagrangian --gpu=2 --mi=0.0 -z=50 -t=mlp &
#nohup python vae_lagrangian.py --lagrangian --gpu=3 --mi=1.0 -z=50 -t=mlp &

python3 vae_lagrangian.py --gpu=1 --mi=-5.0 -z=10  &
python3 vae_lagrangian.py --gpu=2 --mi=0.0 -z=10  &
python3 vae_lagrangian.py --gpu=3 --mi=5.0 -z=10  &
#nohup python vae_lagrangian.py --lagrangian --gpu=1 --mi=-1.0 -z=50  &
#nohup python vae_lagrangian.py --lagrangian --gpu=2 --mi=-1.0 -z=50  &
#nohup python vae_lagrangian.py --lagrangian --gpu=3 --mi=-1.0 -z=50  &