
""" This program expects the path to a batch of experiments in the results directory.
    All output files are placed in a similar subdirectory under analysis.
    Loss and performance logs from all runs are collected and stored in a single
    file each.  For performance, if the files exist, they are not overwritten.

    The program then produces a table of descriptive statistics from the performance
    logs after grouping by one or more columns.  For example, you can see profit
    statistics by agent type, or by learning rate, or by multiple columns at once.
    Check the perf CSV header row to see what is available in your logs.

    It also produces a distribution plot of the first three columns given to -c,
    which are assigned to the column, row, and hue of the grid of plots, and
    separately one column name to -x for the categorical x-axis of each subplot.

    For example, "-c mode date agent -x symbol" will produce a 2-D grid of plots
    by mode (across) and date (down).  Each plot will break out agent type by
    hue, and also draw separate x-axis bars for each symbol.  More than three
    columns can be given to -c for the tabular stats, but will be ignored for
    the plot.

    If loss data exists, the program will also produce loss curves for each model.

    Optionally, a single order book log file can also be specified.  If so, the
    program will produce a plot of the order book and last trade progression for
    one trip within the requested file.  (First trip if no learning agents;
    test_is trip if learning agents.)  If no file is given, it will produce
    a plot using an arbitrary order book log from the requested experiment directory.

    The intervals are determined by the same lobintvl parameter that controls
    the frequency of snapshots to participating market agents.  No cache file
    is written for the order book, as it is already a single file.
"""
import argparse, os
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams.update({'font.size': 12})

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:,.2f}'.format)
pd.set_option('display.multi_sparse', False)


def collect(indir, outdir, kind):
    """ Create a dataframe for the given kind of log from an experiment,
        collecting raw logs as needed. """

    csv = os.path.join(outdir, f"{kind}.csv")

    # If output file exists, just reload it into a dataframe.
    if os.path.exists(csv):
        print (f"Loaded existing {kind} logs from {csv}.")
        return pd.read_csv(csv)

    # Otherwise, collect the raw logs to create the output CSV file.
    print(f"Collecting {kind} logs to {csv}.  Subsequent calls will be faster.")
    files = os.listdir(os.path.join(indir, kind))

    dfs = []
    for file in files:
        f = os.path.join(indir, kind, file)
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(csv, index=False)

    print("Collection complete.")
    return df


### Main program logic starts here.

# Read command-line parameters.
parser = argparse.ArgumentParser(description='Flexible collection/analysis/plotting tool for logs.')

parser.add_argument('-b', '--book', default=None, metavar='ORDER_BOOK_LOG_FILE',
                    help='Name of a single order book log file within the experiment directory.')
parser.add_argument('-c', '--columns', nargs='+', default=['mode','symbol'], metavar='COLUMN_NAMES',
                    help='One or more columns on which to group.  Ex: mode, date, agent.  '
                         'Determines column, row, hue for plots.  Only first three affect plot.')
parser.add_argument('-e', '--exp', required=True, metavar='EXP_RESULTS_PATH',
                    help='Path to experimental results (with raw logs)')
parser.add_argument('-l', '--logs', nargs='+', default=['book','loss','perf'], metavar='LOG_NAMES',
                    help='Process and report/plot only these kinds of logs.')
parser.add_argument('-n', '--every_n', default=1, metavar='BOOK_EVERY_N',
                    help='For book plot, use every Nth update (i.e. lower the frequency for the plot).')
parser.add_argument('-x', '--x_var', default=None, metavar='COLUMN_NAME',
                    help='This column will be the x-axis of each plot.  The y-axis is always profit.  '
                         'Affects plot only.  Optional but usually recommended.')

args = parser.parse_args()

path = args.exp
indir = path[:-1] if path[-1] == "/" else path
outdir = os.path.join("analysis", os.path.basename(indir))
plotdir = os.path.join(outdir, "plot")
if not os.path.isdir(indir) and not os.path.isdir(outdir):
    print (f"Neither {indir} nor {outdir} were found.")
    exit(0)
os.makedirs(plotdir, exist_ok=True)

### Plot and statistical table from performance logs.
if 'perf' in args.logs:
    df_perf = collect(indir, outdir, "perf")

    # Stringify columns that should not get float formatted.
    str_cols = [ x for x in ['lr','tau','trans_cost'] if x in df_perf.columns ]
    for col in str_cols:  df_perf[col] = df_perf[col].astype(str)

    # Print stats distribution grouped by requested columns, showing profit.
    # Usually want to include mode to avoid mixing training and testing results.
    print(df_perf.groupby(args.columns)['profit'].describe())

    # Write the same information to a CSV in the analysis directory.
    df_perf.groupby(args.columns)['profit'].describe().to_csv(os.path.join(outdir, "stats.csv"))

    # Decide which columns to assign to which plot attributes.
    cols = args.columns
    col, row, hue = None, None, None
    match len(cols):
        case 1: col = cols[0]
        case 2: col = cols[0]; row = cols[1]
        case _: col = cols[0]; row = cols[1]; hue = cols[2]

    # Draw, annotate, and save the plot.
    g = sns.catplot(data=df_perf, x=args.x_var, y="profit", col=col, row=row, hue=hue, kind="box",
                    aspect=1.0, margin_titles=True, sharey=False, legend_out=False)
    g.set_ylabels("Daily Profit (USD)")
    g.tick_params(axis='x', rotation=90)
    g.tight_layout()

    plotfile = f"perf_{'_'.join(cols)}.pdf"
    os.makedirs(plotdir, exist_ok=True)

    g.savefig(os.path.join(plotdir, plotfile))


### Plot from loss logs.
if 'loss' in args.logs:
    df_loss = collect(indir, outdir, "loss")

    has_actor = not df_loss['actor_loss'].isnull().all()
    has_critic = not df_loss['critic_loss'].isnull().all()

    if has_actor:
        fig, ax = plt.subplots(figsize=(8,6))
        if has_critic:
            cols = ['actor_loss','critic_loss']
            ax = (df_loss[['global_step']+cols].groupby('global_step').mean(cols)
                        .plot(secondary_y=['critic_loss'], ax=ax))
        else:
            cols = ['actor_loss']
            ax = df_loss[['global_step']+cols].groupby('global_step').mean(cols).plot(ax=ax)
        ax.figure.savefig(os.path.join(plotdir, "loss.pdf"))


### Plot from requested (or arbitrary) order book log file.
if 'book' in args.logs:
    book_file = args.book
    book_path = os.path.join(indir, "book")
    if book_file is None:
        files = os.listdir(book_path)
        book_file = files[0]
    
    book = os.path.join(book_path, book_file)
    if not os.path.isfile(book):
        print (f"Requested order book log file \"{book}\" does not exist.")
        exit()
    
    plot_file = os.path.join(plotdir, "book.pdf")
    
    ### Read book log and combine date and time fields into an index.
    df = pd.read_csv(book)
    print(f"Loaded order book log from {book}.")
    df['timestamp'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['time'], unit='ns')
    df = df.set_index('timestamp')
    
    ### Plot the order book and last trade values.
    fig, ax = plt.subplots(figsize=(8,6))
    levels = df['levels'].iloc[0]
    has_is_test = df['mode'].isin(['test_is']).any()
    df_plot = df.loc[df['mode'] == 'test_is'] if has_is_test else df.loc[df['sim'] == 0]

    ### If requested, resample frequency of book snapshots.
    df_plot = df_plot.iloc[::int(args.every_n)]
    
    ### Plot asks.
    cols = [ f'ask_{i}_p' for i in range(levels) ]
    ax = (df_plot[cols]/100).plot(ax=ax, colormap='autumn')
    
    ### Plot bids.
    cols = [ f'bid_{i}_p' for i in range(levels) ]
    ax = (df_plot[cols]/100).plot(ax=ax, colormap='winter')
    
    ### Plot last trade price.
    ax = (df_plot['trade_price']/100).plot(ax=ax, color='black')
    
    ### Plot fundamental.
    ax = (df_plot['fundamental']/100).plot(ax=ax, color='black', linestyle=':')

    ### Make the plot more attractive.
    hms_only = mdates.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(hms_only)
    ax.get_legend().remove()
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (USD)")
    fig.tight_layout()
    
    ### Save the plot in the analysis directory.
    fig.savefig(plot_file)
    
