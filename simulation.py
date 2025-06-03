
""" This is the main module that controls market simulations.
    It contains the kernel logic and main event loop.

    You should not run it directly.  Copy one of the scripts
    in the experiments directory as your config script,
    which will then call this as appropriate.  You will launch
    the config from the run_exp program.

    This simulation module, as well as non-experimental market
    agents, should use only the args from run_exp, not from
    any specific experiment.
"""

# Basic Python imports and common libraries.
import glob, heapq as hq, numpy as np, os, pickle, sys
from copy import deepcopy
from datetime import datetime
from time import time, time_ns

# Project-specific imports.
from history import History
from market.agent import TradingAgent
from util import ft, latency, TeeOutput


""" Local methods to use internally to the main simulation loop. """

def schedule(dt=None, msg=None, aid=None, a=None, msgid=None):
    """ Schedules a message for delivery to an agent. """
    if dt is None or msg is None or (aid is None and a is None):
        print ("Invalid attempt to schedule.  Simulation terminating.")
        exit()

    if aid is None: aid = a.aid
    if msgid is None:
        msgid = schedule.nextMsg
        schedule.nextMsg += 1

    hq.heappush(schedule.pq, (dt, msgid, aid, msg))



def simulate (sim, model, date, trip, mode, ag_start, end, base_ag, learn_ag, args):
    """ Run simulation of date from agent start time to simulation end time.

        Agents in base_ag are included every day.  If there are learning agents, then
        one agent from learn_ag is included: learn_ag[model].

        Mode can be train, test_is, test_val, or test_oos.  Args is full command line. """

    def log_header (log, extra_cols):
        """ Inner log helper method to generate header row for standard plus extra columns. """
        log.write((",".join([x for x in args.__dict__] +
                        ['sim','model','trip','date','mode','agent','aid'] +
                        extra_cols) + "\n").encode('utf-8'))

    def log_write (log, agent, extra_cols):
        """ Inner log helper method for various logs.  Expects log reference,
            args, agent reference, extra column values as list. """
        log.write((",".join([str(x) for x in list(args.__dict__.values()) +     # config args
                            [sim, model, trip, date, mode] +                    # standard
                            [agent.type, agent.aid] +                           # agent specific
                            extra_cols]) + "\n").encode('utf-8'))               # extra columns

    ft.date_ts = datetime.strptime(date, '%Y-%m-%d').timestamp()
    print(f"Starting simulation: model {model} on {date}, trip {trip}, mode {mode}.")

    # Initialize log header on first simulation.
    if sim == 0:
        ### TEMP HACK: fund to book log
        log_header(simulate.book_log,  ['time'] + [ f'{a}_{i}_{p}' for i in range(args.levels)
                                                    for a in ['ask','bid'] for p in ['p','q'] ] +
                                       ['trade_price','trade_quantity','fundamental'])
        log_header(simulate.loss_log,  ['episode','episode_step','global_step','actor_loss','critic_loss'])
        log_header(simulate.perf_log,  ['mps','profit'])

    # Immutable system properties.
    ex_delay = 1            # currently, exchange never falls behind
    ex_prop = 300           # exchange LOB messages take 300ns to come out
    hist_intvl = 6e10       # play ahead one minute of history to message queue for perf.
    prog_intvl = 3.6e12     # interval to print simulation progress (one hour in ns)

    # Set up the list of agents which will be used today.
    ag = base_ag + ([ learn_ag[model] ] if len(learn_ag) > 0 else [])

    # Call the new_day method on every agent in today's run.
    for a in ag:
        a.new_day(ag_start, evaluate=True if 'test' in mode else False, fixed=args.fixed)

    ### Create data structures that start empty each simulated day.
    pq = []
    schedule.pq = pq
    msg_from_ag = { i:0 for i in range(len(ag)) }
    hist = History(args.symbol, date, args.sim_start, args.datadir)
    cachefile = os.path.join(args.datadir, "cached", f"cache_{args.symbol}_{date}_{ag_start}_"
                                                     f"{args.lobintvl}_{args.seqlen}_{args.levels}")

    # If no history cache file, reconstruct to agent start time and save for future performance.
    if not os.path.isfile(cachefile):
        print (f"Reconstructing history to {ft(ag_start)}")
        hist.reconstruct(ag_start, cachefile, args)

    if not os.path.isfile(cachefile):
        print ("History reconstruction failed.")
        exit()
    else: print ("Reconstruction successful.")

    # Load reconstructed orderbook and fundamental, and fast-forward history.
    with open(cachefile, 'rb') as cache: book = pickle.load(cache)
    book.args = args
    ag[0].book[args.symbol] = book
    hist.fast_forward(ag_start)
    hist.fund = book.fund

    print (f"Loaded book as of {ft(ag_start)} and fast forwarded history.")

    ### Initialize agent times, current time, message ids for this sim
    at = [ a.start for a in ag ]
    ct, schedule.nextMsg = 0, 0
    hist_end = args.sim_start + hist_intvl  # end of current history chunk
    progress = args.exch_open               # reports simulation time progress

    # Seed the queue with a message at exchange open and close times.
    schedule(dt = args.exch_open, aid = 0, msg = { 'type' : 'exopen' })
    schedule(dt = args.exch_close, aid = 0, msg = { 'type' : 'exclose' })

    # Subscribe each trading agent to the order book with requested interval/levels.
    for a in ag:
        if isinstance(a, TradingAgent):
            levels = a.levels if hasattr(a, "levels") else 1
            schedule(dt = a.start, aid = 0, msg = { 'type': 'subscribe', 'aid': a.aid,
                                                    'levels': levels, 'intvl': a.interval })

    # Wallclock start for performance evaluation.
    wcStart = time_ns()

    ### Main simulation loop.
    while len(pq) > 0 and ct <= end:
        # when ct is about to pass hist_end, load next hist_intvl (batched for perf.)
        t, *_ = pq[0]
        if t >= hist_end:
            hist_end += hist_intvl
            for dt,m in hist.history(hist_end):
                schedule(dt = dt, aid = 0, msg = m)
            continue    # in case new hist messages before next agent message

        # heap gets time (pri), msg#, rcp id, msg (dict)
        ct, msgid, rcp, msg = hq.heappop(pq)

        # print periodic progress (after fast-forward)
        while ct > progress:
            if progress > ag_start: print (f"Simulation progress: {ft(progress)} in {(time_ns() - wcStart)/1e9}")
            progress += prog_intvl

        # cannot receive until at <= ct
        if at[rcp] > ct:
            # Don't reschedule LOB snapshots.
            if msg['type'] != 'lob': schedule(dt = at[rcp], aid = rcp, msg = msg, msgid = msgid)
            continue

        # Set agent time to current simulation time.
        at[rcp] = ct

        # When orders are directed towards the exchange, not from history,
        # make a copy on passthrough to ensure agent local orders do not change.
        if rcp == 0 and 'order' in msg and msg['order'].aid > 0: msg['order'] = deepcopy(msg['order'])

        # Deliver the message and retrieve any number of (rcp, msg) tuples in reply.
        wc = time_ns()
        for r,m in ag[rcp].message(ct, msg):
            at[rcp] = ct + (ex_delay if rcp == 0 else time_ns() - wc)

            # Process new message.  Latency determined by non-exchange agent.
            if r == 0: minlat = ag[rcp].minlat
            else: minlat = ag[r].minlat

            # Determine delivery time and schedule message.
            dt = at[rcp] + latency(minlat) + (ex_prop if rcp == 0 else 0)
            schedule(dt = dt, aid = r, msg = m)

            msg_from_ag[rcp] += 1

        # Delay the agent that just acted according to its total processing time.
        at[rcp] = ct + (ex_delay if rcp == 0 else time_ns() - wc)

        # Allow the agent to record training loss if it wishes.
        if msg['type'] == 'lob':
            extra_cols = ag[rcp].report_loss()
            if extra_cols is not None: log_write(simulate.loss_log, ag[rcp], list(extra_cols))

        # After processing order messages to the exchange, allow order book logging if needed.
        if rcp == 0 and 'order' in msg:
            extra_cols = ag[rcp].book[args.symbol].log_book()
            if extra_cols is not None: log_write(simulate.book_log, ag[rcp], extra_cols)

    # Get wallclock elapsed time for performance. 
    wcElapsed = time_ns() - wcStart

    elapsed = wcElapsed / 1e9
    mps = msgid / elapsed
    # Could count queued or dequeued messages.  Counting dequeued.
    print (f"Processed {msgid} messages in {elapsed} seconds.  MPS: {mps}")

    # Close history file.
    hist.close()

    # Report number of messages sent by each agent.
    print (f"msg_from_ag: {msg_from_ag}")

    # Final agent accounting.  Exchange is always agent 0.
    fin_book = ag[0].book[args.symbol]
    fin_snap, fin_fund, fin_hist = fin_book.snap, fin_book.fund, fin_book.snap_hist

    # Allow all agents to make final computations and log results.
    r, c = {}, {}
    for a in base_ag + ([learn_ag[model]] if len(learn_ag) > 0 else []):
        if a.aid == 0: continue        # No "results" for exchange

        profit = a.finalize_episode(ct, { 'type': 'lob', 'snap': fin_snap,
                                          'fund': fin_fund, 'hist': fin_hist })
        log_write(simulate.perf_log, a, [mps, profit])

        # Also collect profit by agent type.
        tag = a.type + a.tag            # tag allows distinguishing flavors of same agent type
        r[tag], c[tag] = r.get(tag,0) + profit, c.get(tag,0) + 1

    # Report mean profit by agent type.
    for t in sorted(r):
        pnl = r[t]/c[t]
        print (f"sim {sim} model {model} trip {trip} ({date}:{mode}), {t:30} profit: {pnl:>20.2f}")
    
    print (f"Simulation {sim} ending!")


def run_experiment(base_agents, learning_agents, args):
    """ Call this method from your experiment file to initiate
        a complete train-val-test cycle over all requested simulations.
        Expects util.args to contain at least the basic arguments from run_exp.

        Base agents are present every day.  Learning agents are trained
        one at a time, round-robin, before being validated/tested. """
    
    # Create loss, out, and perf subdirectories within the batch-specific results directory.
    for sub in ['book','loss','out','perf']:
        os.makedirs(f"{args.result_dir}/{sub}", exist_ok=True)
    sys.stdout = TeeOutput(f"{args.result_dir}/out/{args.runtag}")
    simulate.book_log = open(f"{args.result_dir}/book/{args.runtag}", "wb", buffering=0)
    simulate.loss_log = open(f"{args.result_dir}/loss/{args.runtag}", "wb", buffering=0)
    simulate.perf_log = open(f"{args.result_dir}/perf/{args.runtag}", "wb", buffering=0)
    
    print("Random seed:", args.seed)

    ### NO LEARNING AGENTS

    # If there are no learning agents, training, validation, and testing aren't meaningful.
    # We use the "training" parameters for consistency with other experiments.
    if len(learning_agents) == 0:
        sim = -1
        for trip in range(args.trips):
            for date in args.train_dates:
                sim += 1
                simulate(sim, 'n/a', date, trip, 'simulate', args.train_start,
                         args.train_end, base_agents, learning_agents, args)

    else:
        ### TRAINING and IN-SAMPLE TESTING
    
        # Train each model on each training day.  Repeat the full list of days "trips" times.
        # Then perform one in-sample test trip on the same data.
        sim = -1
        for trip in range(args.trips + 1):
            for model in range(len(learning_agents)):
                for date in args.train_dates:
                    sim += 1
                    mode = 'test_is' if trip == args.trips else 'train'
                    simulate(sim, model, date, trip, mode, args.train_start,
                             args.train_end, base_agents, learning_agents, args)
    
        ### VALIDATION
    
        # Evaluate each model on validation period once.  No model selection yet.
        trip += 1
        for model in range(len(learning_agents)):
            for date in args.val_dates:
                sim += 1
                simulate(sim, model, date, trip, 'test_val', args.val_start,
                         args.val_end, base_agents, learning_agents, args)
    
        ### OUT-OF-SAMPLE TESTING
    
        # Evaluate each model on test period once.
        # (Once model selection in validation, evaluate only "best" model.)
        trip += 1
        for model in range(len(learning_agents)):
            for date in args.test_dates:
                sim += 1
                simulate(sim, model, date, trip, 'test_oos', args.test_start,
                         args.test_end, base_agents, learning_agents, args)

    print (f"All done: {args.tag}.")

    simulate.book_log.close()
    simulate.loss_log.close()
    simulate.perf_log.close()

