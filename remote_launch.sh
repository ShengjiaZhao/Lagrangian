#!/usr/bin/env bash
python vae_lagrangian.py --gpu=0 --mi=-10.0 -z=10 --slack=$1 &
python vae_lagrangian.py --gpu=1 --mi=-5.0 -z=10 --slack=$1 &
python vae_lagrangian.py --gpu=2 --mi=-2.0 -z=10 --slack=$1 &
python vae_lagrangian.py --gpu=3 --mi=-1.0 -z=10 --slack=$1 &
python vae_lagrangian.py --gpu=0 --mi=1.0 -z=10 --slack=$1 &
python vae_lagrangian.py --gpu=1 --mi=2.0 -z=10 --slack=$1 &
python vae_lagrangian.py --gpu=2 --mi=5.0 -z=10 --slack=$1 &
python vae_lagrangian.py --gpu=3 --mi=10.0 -z=10 --slack=$1 &