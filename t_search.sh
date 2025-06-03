#!/bin/bash
#SBATCH --output=results/.slurm/%x-%j.out  # Output file
#SBATCH --error=results/.slurm/%x-%j.err   # Error file

### Adapt to your cluster configuration.
#SBATCH --nodes=6
#SBATCH --ntasks=192
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5GB

### Can handle any number of permutations and run ntasks at a time,
### queuing automatically, so no parameters needed except an identifier.
### Just put everything in this script's for loops.

### Additional search hyperparameters can be added by following the
### example of those present.  The cross product of all combinations
### of parameters will be run a number of times equal to the repeat loop.

if [ $# -lt 1 ]; then
  echo $0: 'usage: sbatch <this_script> <user_tag>'
  exit 1
fi

usertag="$1"

# Source your virtual environment.
source ./venv/bin/activate

# Initialize task num for separate log file generation.
tasknum=1

# Record a single timestamp for the entire batch.
ts=$EPOCHSECONDS

### To avoid race conditions, when running a new date or time
### for the first time, consider a single run per combination
### of symbol + date + time, with a single training trip to
### cache all new data files.

### Multiple values can be given as a range, e.g.
### for value in {5..15}; do
### or as a list, e.g.
### for level in 10 20 30; do

# No indentation due to extremely deep nesting.

for repeat in {1..42}; do	# ALL: how many times to repeat every hp combination

for date in 2019-12-01; do	# ALL: assumes train, val, test on one date
for level in 10; do		# ALL: LOB levels to offer agents (and log)
for lobintvl in 15e9; do	# ALL: how often to update LOB snapshots (and log)
for seqlen in 20; do		# ALL: how many recent LOB snapshots to offer agents
for symbol in AAPL; do		# ALL: one symbol per simulation
for trip in 1; do		# ALL: trips through training period

for train_start in 9.5; do	# ALL: train start each day in hours since midnight
for train_end in 16.0; do	# ALL: train end each day in hours since midnight
for test_start in 9.5; do	# ALL: test start each day in hours since midnight
for test_end in 16.0; do	# ALL: test end each day in hours since midnight
for val_start in 9.5; do	# ALL: val start each day in hours since midnight
for val_end in 16.0; do		# ALL: val end each day in hours since midnight

for bg_mkt in 0 1; do		# BG: how many market makers?
for bg_mom in 0 1 5 10 25; do	# BG: how many momentum traders?
for bg_nse in 0 10 25; do	# BG: how many noise traders?
for bg_obi in 0; do		# BG: how many order book imbalance traders?
for bg_val in 0 1 5 10 25; do	# BG: how many value traders?

for rlagent in none; do		# RL: which learning agent to use
for embed in 3; do		# RL: embedding size for encoder
for encoder in lstm; do		# RL: which sequence encoder to use
for lr in 5e-1; do		# RL: learning rate
for model in 1; do		# RL: how many models to alternate while training
for tau in 0.2; do		# RL: target network rate of drift towards actor network
for trade in 10; do		# RL: maximum size of one order
for transcost in 0.00; do	# RL: transaction cost (fraction of transaction value)

for expnoise in 0.5; do		# TD3: exploration noise
for polnoise in 0.5; do		# TD3: policy noise
for polfreq in 10; do		# TD3: policy update frequency (in steps)

for trainfreq in 5; do		# DQN: training frequency (in steps)
for targetfreq in 40; do	# DQN: target update frequenct (in steps)

    # Example reinforcement learning experiment.
    srun -u --exclusive=user --mem-per-cpu 5GB -N1 -n1 /usr/bin/time -v python -u run_exp.py --exp rl --rlagent $rlagent --expnoise $expnoise --polnoise $polnoise --polfreq $polfreq --train_freq $trainfreq --target_freq $targetfreq --train_dates $date --val_dates $date --test_dates $date --train_start $train_start --train_end $train_end --val_start $val_start --val_end $val_end --test_start $test_start --test_end $test_end --symbol $symbol --bg_mkt $bg_mkt --bg_mom $bg_mom --bg_nse $bg_nse --bg_obi $bg_obi --bg_val $bg_val --lr $lr --fixed --models $model --trips $trip --seqlen $seqlen --lobintvl $lobintvl --levels $level --trade $trade --encoder $encoder --embed $embed --tau $tau --trans_cost $transcost --ts $ts --tag $usertag --runtag "${symbol}_${repeat}_${tasknum}" &

    tasknum=$((tasknum + 1))

done; done; done; done; done; done; done; done; done; done
done; done; done; done; done; done; done; done; done; done
done; done; done; done; done; done; done; done; done; done
done

wait
echo Done.
