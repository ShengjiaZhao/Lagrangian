#!/usr/bin/env bash
python vae_lagrangian.py --lagrangian --gpu=1 --mi=-5.0 &
python vae_lagrangian.py --lagrangian --gpu=1 --mi=-1.0 &
python vae_lagrangian.py --lagrangian --gpu=2 --mi=0.0 &
python vae_lagrangian.py --lagrangian --gpu=2 --mi=0.5 &
python vae_lagrangian.py --lagrangian --gpu=3 --mi=1.0 &
python vae_lagrangian.py --lagrangian --gpu=3 --mi=5.0 &