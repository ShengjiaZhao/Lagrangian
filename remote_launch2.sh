#!/usr/bin/env bash
python vae_lagrangian2.py --lagrangian --gpu=0 --mi=-0.5 -z=10  &
python vae_lagrangian2.py --lagrangian --gpu=1 --mi=-0.2 -z=10  &
python vae_lagrangian2.py --lagrangian --gpu=2 --mi=-0.1 -z=10  &
python vae_lagrangian2.py --lagrangian --gpu=3 --mi=0.0 -z=10  &
python vae_lagrangian2.py --lagrangian --gpu=1 --mi=0.1 -z=10  &
python vae_lagrangian2.py --lagrangian --gpu=2 --mi=0.2 -z=10  &
python vae_lagrangian2.py --lagrangian --gpu=3 --mi=0.5 -z=10  &