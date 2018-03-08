#!/usr/bin/env bash
python vae_lagrangian2.py --lagrangian --gpu=0 --mi=-10.0 -z=5  &
python vae_lagrangian2.py --lagrangian --gpu=1 --mi=-5.0 -z=5  &
python vae_lagrangian2.py --lagrangian --gpu=2 --mi=-1.0 -z=5  &
python vae_lagrangian2.py --lagrangian --gpu=3 --mi=0.0 -z=5  &
python vae_lagrangian2.py --lagrangian --gpu=0 --mi=1.0 -z=5  &
python vae_lagrangian2.py --lagrangian --gpu=1 --mi=5.0 -z=5  &
python vae_lagrangian2.py --lagrangian --gpu=2 --mi=10.0 -z=5  &
python vae_lagrangian2.py --lagrangian --gpu=3 --mi=0.5 -z=5  &