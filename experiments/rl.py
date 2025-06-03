
""" This is an example experiment in which a reinforcement learning
    agent with continuous internal and environmental observations
    interacts with an LOB populated by real historical orders.
    Depending on the configuration, it may be trading interactively
    against other trading agents as well.  Multiple RL algorithms
    are supported, featuring both discrete and continuous actions.

    The experiment should not be directly run.  Rather, it should be
    passed as the --exp parameter to run_exp.py in the main project
    directory.

    Also note that only experiment-specific arguments are parsed in
    this file.  There are many basic arguments required for all experiments,
    which are handled in run_exp.py before handing control to this file.
    It is not necessary to add every agent parameter to argparse, only those
    that may be part of a hyperparameter search.

    It is strongly suggested to copy a working experiment as the model
    for any new experiments.
"""

if __name__ == "__main__":
    print("This file cannot be run directly.  Please pass it to the --exp argument "
          "of run_exp.py in the main project directory.")
    exit()


def experiment(base_args, exp_args):

    # Built-in or PyPy imports.
    import argparse, numpy as np

    # Custom imports from this project.
    from market.background import MarketMakerAgent, MomentumAgent, NoiseAgent, \
                                  OrderBookImbalanceAgent, ValueAgent
    from market.exchange import ExchangeAgent
    from rl.ddpg_td3 import DDPGAgent
    from rl.dqn import DQNAgent
    from simulation import run_experiment

    # Read experiment-specific command-line parameters.
    parser = argparse.ArgumentParser(description='This configuration is for a deep RL intraday trading task.',
                                     add_help=False)
    
    # Background agent options for this experiment.
    bg_group = parser.add_argument_group('Background agent options for the RL experiment.')
    bg_group.add_argument('--bg_mkt', type=int, default=0, help="Number of market makers.")
    bg_group.add_argument('--bg_mom', type=int, default=0, help="Number of momentum traders.")
    bg_group.add_argument('--bg_nse', type=int, default=0, help="Number of noise traders.")
    bg_group.add_argument('--bg_obi', type=int, default=0, help="Number of OBI traders.")
    bg_group.add_argument('--bg_val', type=int, default=0, help="Number of value traders.")

    # RL agent arguments specific to this experiment (for hyperparameter search)
    # but common to all supported RL algorithms.
    rl_group = parser.add_argument_group('Basic RL parameters.')
    rl_group.add_argument('--rlagent', default="dqn", help="RL agent: dqn, ddpg, none, td3.")
    rl_group.add_argument('--rldebug', action="store_true", help="Print more detailed RL info.")

    # Order quantity control.
    quantity_group = parser.add_argument_group('Order quantity control parameters.')
    quantity_group.add_argument('--trade', type=int, default=10, metavar='INT',
                                help='Maximum size of single trade action.')
    quantity_group.add_argument('--shares', type=int, default=100, metavar='INT',
                                help='Maximum position size (long or short).')

    # Observation preprocessing.
    pre_group = parser.add_argument_group('Environmental observation preprocessing parameters.')
    pre_group.add_argument('--encoder', default="lstm", help="Encode env obs sequences with: lstm, none.")
    pre_group.add_argument('--embed', type=int, default=6, metavar='INT',
                           help='Embed size for environmental observations (if encoder selected).')

    # Standard Deep RL and/or neural network parameters.
    net_group = parser.add_argument_group('Standard DeepRL/network parameters.')
    net_group.add_argument('--batchsize', type=int, default=32, metavar='INT',
                           help='Batch size for replay training.')
    net_group.add_argument('--gamma', type=float, default=0.99, metavar='FLOAT',
                           help='Gamma for Bellman equations.')
    net_group.add_argument('--lr', type=float, default=5e-2, metavar='FLOAT',
                           help='Learning rate for agent/policy optimizers.')
    net_group.add_argument('--netsize', type=int, default=64, metavar='INT',
                           help='Network size for agent/policy hidden layers.')
    net_group.add_argument('--rbuf', type=int, default=10000, metavar='INT',
                           help='Size of replay buffer.')
    net_group.add_argument('--tau', type=float, default=1.0, metavar='FLOAT',
                           help='Interpolation coefficient for target network updates.')

    # Parameters relating to the RL agent's participation in the simulation.
    sim_group = parser.add_argument_group('Simulation participation parameters.')
    sim_group.add_argument('--interval', type=float, default=30e9, metavar='FLOAT',
                           help='Approx. ns between agent actions.')
    sim_group.add_argument('--latency', type=float, default=1e1, metavar='FLOAT',
                           help='Latency for RL agent (ns).')
    sim_group.add_argument('--models', type=int, default=1, metavar='INT',
                           help='How many models to train and select from in validation.')
    sim_group.add_argument('--startstep', type=int, default=5, metavar='FLOAT',
                           help='Random actions before learning starts.')
    sim_group.add_argument('--trans_cost', type=float, default=0.0005, metavar='INT',
                           help='Transaction cost by value of transaction (0.0001 == 1 basis point).')

    # Parameters only applicable to DDPG and TD3.
    ddpg_group = parser.add_argument_group('Parameters specific to DDPG and TD3.')
    ddpg_group.add_argument('--expnoise', type=float, default=0.1, metavar='FLOAT',
                            help='Exploration noise for actor.')
    ddpg_group.add_argument('--polfreq', type=int, default=2, metavar='INT',
                            help='Interval for policy updates.')
    ddpg_group.add_argument('--polnoise', type=float, default=0.4, metavar='FLOAT',
                            help='Policy noise for q-network.')

    # Parameters only applicable to DQN.
    dqn_group = parser.add_argument_group('Parameters specific to DQN.')
    dqn_group.add_argument('--start_e', type=float, default=1.0, metavar='FLOAT',
                           help='Starting exploration rate (epsilon).')
    dqn_group.add_argument('--end_e', type=float, default=0.05, metavar='FLOAT',
                           help='Ending exploration rate (epsilon).')
    dqn_group.add_argument('--explore_frac', type=float, default=0.8, metavar='FLOAT',
                           help='Fraction of total training actions to reach end_e.')
    dqn_group.add_argument('--train_freq', type=int, default=5, metavar='INT',
                           help='Interval for training updates.')
    dqn_group.add_argument('--target_freq', type=int, default=40, metavar='INT',
                           help='Interval for target updates.')

    # Parse the remaining experimental arguments not handled by run_exp.
    # Combine the basic and experimental arguments to a single namespace.
    exp_args = parser.parse_args(exp_args)
    args = argparse.Namespace(**vars(base_args), **vars(exp_args))

    # Configure the base agents for the experiment.  Exchange must be agent zero.
    # "Base" agents are those always present for this experiment.

    base_ag = [ ExchangeAgent([args.symbol], args) ] + \
              [ MarketMakerAgent(args.symbol, 1e1, 1e8, 100) for i in range(args.bg_mkt)] + \
              [ MomentumAgent(args.symbol, 5e6, 1e9, 100) for i in range(args.bg_mom)] + \
              [ NoiseAgent(args.symbol, 12e6, 1e9) for i in range(args.bg_nse)] + \
              [ OrderBookImbalanceAgent(args.symbol, 1e3, 2e8, 100, 50) for i in range(args.bg_obi)] + \
              [ ValueAgent(args.symbol, 3e6, 1e9, 100) for i in range(args.bg_val)]

    # Configure the learning agents for the experiment.  They are separated out
    # because only one is included in each simulated day.  (args.models models
    # are created as separate agents and round-robined for training, validation,
    # and testing)
    args.rlagent = args.rlagent.lower()
    learn_ag = []

    if args.rlagent != 'none':
        # DQN epsilon decay requires an estimate of total actions taken during training.
        args.total_timesteps = 0
        if args.rlagent == 'dqn':
            args.total_timesteps = (args.train_end - args.train_start) * args.trips / args.interval
            print("Estimated total timesteps is", args.total_timesteps)
    
        # Create as many separate models as the experiment requested.
        for _ in range(args.models):
            # Observation extents are the same for all Deep RL agents.
            obs_low = np.array([[ -1.0, -2.0 ] + [ 0.0 ] * (args.levels * 2)] * args.seqlen)
            obs_high = np.array([[ 1.0, 2.0 ] + [ 1.0 ] * (args.levels * 2)] * args.seqlen)
    
            # Configure discrete or continuous action space as needed and instantiate RL agent.
            if args.rlagent == 'dqn':
                num_act = 3
                a = DQNAgent(args, obs_low, obs_high, num_act)
            elif args.rlagent in ['ddpg','td3']:
                act_low, act_high = np.array([-2.0]), np.array([2.0])
                a = DDPGAgent(args, obs_low, obs_high, act_low, act_high)
            else:
                print (f"Unknown RL agent: {args.rlagent}.")
                exit()
    
            # These models must all share one agent id, since they will be used one at a time,
            # substituting for each other.  All base agents must be added before learning agents.
            a.aid = len(base_ag)
            learn_ag.append(a)
    
    # Launch the experiment.
    run_experiment(base_ag, learn_ag, args)
    
