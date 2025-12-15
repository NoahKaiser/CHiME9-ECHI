#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=Enhance-Baseline
#SBATCH --output=job_name%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=7-00:00:00
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --qos=batch
#SBATCH --nodelist=linse3

# Activate everything you need
#module load cuda/12.8
#module load cuda #versuche, ob 11.8 auch verfuegbar auf Server
#pyenv activate venv
# Run your python code

#uv run --extra=cu128 run_train.py --config-name main_ha
#uv run --extra=cu128 run_enhancement.py resample.run=false #resampling only has to be done once
#uv run --extra=cu118 DataPreProcess/process_echoset.py --in_dir /data/public/EchoSet --out_dir out
#uv run --extra=cu128 check_cuda_status.py
#uv run --extra=cu128 check_checkpoints.py
#uv run --extra=cpu run_evaluation.py
uv run visualize_reports_matplotlib.py \
  --reports_dir "/no_backups/s1495/experiments/baseline_1/evaluation/reports" \
  --metrics pysepm_fwsegsnr sdr sar si_snr ci_sdr\
  --dpi=300\
  --filter_signal_type="summed"
#nvidia-smi
#module list