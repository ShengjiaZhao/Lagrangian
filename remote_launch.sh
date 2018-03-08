#!/usr/bin/env bash
python vae_lagrangian2.py --lagrangian --gpu=0 --mi=-10.0 -z=10  &
python vae_lagrangian2.py --lagrangian --gpu=1 --mi=-5.0 -z=10  &
python vae_lagrangian2.py --lagrangian --gpu=2 --mi=-2.0 -z=10  &
python vae_lagrangian2.py --lagrangian --gpu=3 --mi=-1.0 -z=10  &
python vae_lagrangian2.py --lagrangian --gpu=0 --mi=1.0 -z=10  &
python vae_lagrangian2.py --lagrangian --gpu=1 --mi=2.0 -z=10  &
python vae_lagrangian2.py --lagrangian --gpu=2 --mi=5.0 -z=10  &
python vae_lagrangian2.py --lagrangian --gpu=3 --mi=10.0 -z=10  &