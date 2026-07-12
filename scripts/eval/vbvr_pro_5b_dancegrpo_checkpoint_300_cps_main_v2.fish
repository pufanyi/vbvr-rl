#!/usr/bin/env fish

set -lx CHECKPOINT_STEP 300
exec fish (dirname (status filename))/vbvr_pro_5b_dancegrpo_checkpoint_cps_main_v2.fish $argv
