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

python vae_lagrangian2.py --lagrangian --gpu=1 --mi=-5.0 -z=5  &
python vae_lagrangian2.py --lagrangian --gpu=2 --mi=0.0 -z=5  &
python vae_lagrangian2.py --lagrangian --gpu=3 --mi=5.0 -z=5  &
#nohup python vae_lagrangian.py --lagrangian --gpu=1 --mi=-1.0 -z=50  &
#nohup python vae_lagrangian.py --lagrangian --gpu=2 --mi=-1.0 -z=50  &
#nohup python vae_lagrangian.py --lagrangian --gpu=3 --mi=-1.0 -z=50  &