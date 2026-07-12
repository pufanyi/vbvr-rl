#!/usr/bin/env fish
set -lx CHECKPOINT_STEP 2100
exec fish (dirname (status filename))/vbvr_pro_5b_dancegrpo_checkpoint_main_v2.fish $argv
