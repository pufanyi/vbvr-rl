#!/usr/bin/env fish

source (dirname (status filename))/../../lib/env.fish

pixi run --locked python -m src.cli.summarize_vbvr_pro_results $argv
