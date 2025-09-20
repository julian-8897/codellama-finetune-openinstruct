#!/bin/bash
export WANDB_PROJECT=codellama-openinstruct-project
export WANDB_RUN_NAME=codellama-openinstruct-run  # optional, overrides run_name in TrainingArguments

python main.py