#!/usr/bin/env fish
set -lx CHECKPOINT_STEP 2400
set -lx CPS_NOISE_LEVEL 0.3
exec fish (dirname (status filename))/vbvr_pro_5b_dancegrpo_checkpoint_cps_main_v2.fish $argv
