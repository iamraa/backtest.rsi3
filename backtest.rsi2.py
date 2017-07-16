"""
RSI(2) multithread optimizator

2017-07-11
"""
import sys
from datetime import datetime

from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
import logbook
import pandas as pd
import pytz
from zipline.algorithm import TradingAlgorithm
from zipline.api import symbol, record, order_target_percent, set_benchmark
from zipline.finance.trading import SimulationParameters
from zipline.utils.calendars import get_calendar

import numpy as np
import talib

from utils.load_local import load

zipline_logging = logbook.NestedSetup([
    logbook.NullHandler(),
    logbook.StreamHandler(sys.stdout, level=logbook.INFO),
    logbook.StreamHandler(sys.stderr, level=logbook.ERROR),
])
zipline_logging.push_application()


# code
def initialize(context):
    context.set_benchmark(context.symbol('SPY'))
    context.asset = context.symbol('SPY')
    # """
    context.RSI_PERIOD = 3
    context.RSI_BOTTOM, context.RSI_TOP, context.RSI_SHIFT = 30, 70, 15
    context.MAX_LONG, context.MAX_SHORT = 5, 15
    context.PROB_WINDOW = 10
    context.MIN_PROB = 20.
    # """
    context.MA_SHORT, context.MA_LONG = 50, 200
    context.days_ago = 0

    # add progressbar
    context.total_days = (
        context.sim_params.end_session - context.sim_params.start_session).days
    context.pbar = tqdm(total=context.total_days)


# code
def handle_data(context, data):
    if not data.can_trade(context.asset):
        print("Can't trade {0}".format(context.asset.symbol))
        return

    prices = data.history([context.asset], 'price', 1000, '1d')
    ma_short = talib.SMA(prices[context.asset].values,
                         timeperiod=context.MA_SHORT)
    ma_long = talib.SMA(prices[context.asset].values,
                        timeperiod=context.MA_LONG)
    rsi = talib.RSI(prices[context.asset].values, timeperiod=context.RSI_PERIOD)

    is_bear_market = ma_short[-1] < ma_long[-1]
    is_bull_market = not is_bear_market

    # shift thresholds
    rsi_top = context.RSI_TOP + (context.RSI_SHIFT if is_bull_market else 0)
    rsi_bottom = context.RSI_BOTTOM - (
    context.RSI_SHIFT if is_bear_market else 0)

    window = context.RSI_PERIOD * context.PROB_WINDOW
    probability_bottom = probability(
        rsi[-window:], thresholds=(context.RSI_BOTTOM, context.RSI_TOP),
        what='bottom')

    # is_bull_market = True
    # probability_bottom = 50

    # Trading logic # and probability_bottom >= context.MIN_PROB
    if is_bull_market and probability_bottom >= context.MIN_PROB:
        if rsi[-2] > rsi_bottom and rsi[-1] < rsi_bottom:
            # buy
            order_target_percent(context.asset, 1.)
            context.days_ago = 0
        elif rsi[-2] < rsi_top and rsi[-1] > rsi_top:
            order_target_percent(context.asset, 0)
    elif is_bear_market:
        order_target_percent(context.asset, 0)

    open_total = np.sum(
        [x.amount * x.cost_basis for x in context.portfolio.positions.values()])
    if open_total:
        context.days_ago += 1

    if context.days_ago > context.MAX_LONG and not context.get_open_orders():
        order_target_percent(context.asset, 0)
        context.days_ago = 0

    # Save values for later inspection
    record(price=data.current(context.asset, 'price'),
           ma_short=ma_short[-1],
           ma_long=ma_long[-1],
           rsi=rsi[-1],
           rsi_top=rsi_top,
           rsi_bottom=rsi_bottom,
           open_total=open_total,
           probability_bottom=probability_bottom
           )

    if context.pbar:
        # update progressbar
        days = (context.get_datetime() - context.sim_params.start_session).days
        if days > context.pbar.n:
            context.pbar.update(days - context.pbar.n)


def rle(inarray):
    """
    Run length encoding. Partial credit to R rle function.
    Multi datatype arrays catered for including non Numpy

    Wiki: http://bit.ly/wiki_rle
    Source: htts://stackoverflow.com

    returns:
        tuple (runlengths, startpositions, values)
    """
    ia = np.array(inarray)  # force numpy
    n = len(ia)
    if n == 0:
        return (None, None, None)
    else:
        y = np.array(ia[1:] != ia[:-1])  # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)  # must include last element posi
        z = np.diff(np.append(-1, i))  # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        return (z, p, ia[i])


def probability(y, thresholds=(30, 70), what='top'):
    """
    Calculate reverse probability
    """
    if what == 'top':
        runlengths, startindices, values = rle(y >= thresholds[1])
        if runlengths is None:
            return 0
        above = runlengths[values]
        return len(above) / sum(above) * 100 if len(above) else 0
    elif what == 'bottom':
        runlengths, startindices, values = rle(y <= thresholds[0])
        if runlengths is None:
            return 0
        below = runlengths[values]
        return len(below) / sum(below) * 100 if len(below) else 0
    else:
        return 0


def analyze(context, perf):
    if context.pbar:
        # close progressbar, if use
        context.pbar.close()

    # Results
    print(
        "Returns: {0:.2f}% MaxDrawdown: {1:.2f}% Benchmark: {2:.2f}%".format(
            perf.algorithm_period_return[-1] * 100,
            perf.max_drawdown[-1] * 100,
            perf.benchmark_period_return[-1] * 100
        ))

    fig = plt.figure(figsize=(15, 10), facecolor='white')

    ax1 = fig.add_subplot(411)
    perf.portfolio_value.plot(ax=ax1)
    ax1.set_ylabel('portfolio value in $')

    ax2 = fig.add_subplot(412)
    perf['price'].plot(ax=ax2)
    perf[['ma_short', 'ma_long']].plot(ax=ax2)

    perf_trans = perf.ix[[t != [] for t in perf.transactions]]
    buys = perf_trans.ix[
        [t[0]['amount'] > 0 for t in perf_trans.transactions]]
    sells = perf_trans.ix[
        [t[0]['amount'] < 0 for t in perf_trans.transactions]]
    ax2.plot(buys.index, perf.price.ix[buys.index],
             '^', markersize=5, color='m')
    ax2.plot(sells.index, perf.price.ix[sells.index],
             'v', markersize=5, color='k')
    ax2.set_ylabel('price in $')

    ax3 = fig.add_subplot(413)
    perf['rsi'].plot(ax=ax3)
    perf['rsi_top'].plot(ax=ax3)
    perf['rsi_bottom'].plot(ax=ax3)
    perf['probability_bottom'].plot(ax=ax3, color='r')
    ax3.axhline(50, ls='-.', lw=0.5, color='k')
    ax3.plot(buys.index, perf.rsi.ix[buys.index],
             '^', markersize=6, color='m')
    ax3.plot(sells.index, perf.rsi.ix[sells.index],
             'v', markersize=6, color='k')
    # ax3.axhline(context.RSI_TOP)
    # ax3.axhline(context.RSI_BOTTOM)

    ax4 = fig.add_subplot(414)
    perf['open_total'].plot(ax=ax4)
    ax4.axhline(0)

    plt.legend(loc=0)
    plt.show()

# Load data manually
start = datetime(2000, 1, 1, 0, 0, 0, 0, pytz.utc).date()
end = datetime(2017, 7, 1, 0, 0, 0, 0, pytz.utc).date()
data = load(['SPY'], start=start, end=end)

"""
Prepare simulation
"""
sim_params = SimulationParameters(
    start_session=pd.Timestamp('2004-01-01', tz=pytz.utc),
    end_session=pd.Timestamp('2017-07-01', tz=pytz.utc),
    capital_base=1.0e5,
    data_frequency='daily',
    trading_calendar=get_calendar("NYSE"),
)

period = [3]
bottom = [30]
top = [70]
shift = [15]
max_long = [5]
prob_window = [10]
prob = [20]

def worker(args):
    """thread worker function"""
    period, bottom, top, shift, max_long, prob_window, prob = args

    algo_obj = TradingAlgorithm(
        initialize=initialize,
        handle_data=handle_data,
        analyze=analyze,
        sim_params=sim_params
    )

    algo_obj.RSI_PERIOD = period
    algo_obj.RSI_BOTTOM, algo_obj.RSI_TOP, algo_obj.RSI_SHIFT = bottom, top, shift
    algo_obj.MAX_LONG, algo_obj.MAX_SHORT = max_long, max_long
    algo_obj.PROB_WINDOW = prob_window
    algo_obj.MIN_PROB = prob

    perf = algo_obj.run(data, overwrite_sim_params=False)
    print(
        ("Returns: {0:.2f}% MaxDrawdown: {1:.2f}% " +
         "Benchmark: {2:.2f}% Args: {3}").format(
            perf.algorithm_period_return[-1] * 100,
            perf.max_drawdown[-1] * 100,
            perf.benchmark_period_return[-1] * 100,
            args
        ))

    return


worker(*list(itertools.product(
    period, bottom, top, shift, max_long, prob_window, prob)))

