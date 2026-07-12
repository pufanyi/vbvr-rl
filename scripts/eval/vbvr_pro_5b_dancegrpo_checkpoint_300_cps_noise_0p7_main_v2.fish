#!/usr/bin/env fish
set -lx CPS_NOISE_LEVEL 0.7
exec fish (dirname (status filename))/vbvr_pro_5b_dancegrpo_checkpoint_300_cps_main_v2.fish $argv
