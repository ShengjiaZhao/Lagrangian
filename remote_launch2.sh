#!/usr/bin/env bash
python vae_lagrangian.py --gpu=0 --mi=-0.5 -z=10 --slack=$1 &
python vae_lagrangian.py --gpu=1 --mi=-0.2 -z=10 --slack=$1 &
python vae_lagrangian.py --gpu=2 --mi=-0.1 -z=10 --slack=$1 &
python vae_lagrangian.py --gpu=3 --mi=0.0 -z=10 --slack=$1 &
python vae_lagrangian.py --gpu=1 --mi=0.1 -z=10 --slack=$1 &
python vae_lagrangian.py --gpu=2 --mi=0.2 -z=10 --slack=$1 &
python vae_lagrangian.py --gpu=3 --mi=0.5 -z=10 --slack=$1 &