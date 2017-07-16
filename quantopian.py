"""
Quantrum: RSI(2)

"""

import talib
import numpy as np


# Setup our variables
def initialize(context):
    set_benchmark(symbol('SPY'))
    context.asset = symbol('SPY')

    context.RSI_PERIOD = 3
    context.RSI_BOTTOM, context.RSI_TOP, context.RSI_SHIFT = 30, 70, 15
    context.MAX_LONG, context.MAX_SHORT = 5, 15
    context.PROB_WINDOW = 10
    context.MIN_PROB = 20.

    context.MA_SHORT, context.MA_LONG = 50, 200

    context.days_ago = 0

    schedule_function(rsi_reverse, date_rules.every_day(), time_rules.market_close())

def rsi_reverse(context, data):
    if not data.can_trade(context.asset):
        print("Can't trade {0}".format(context.asset.symbol))
        return

    prices = data.history([context.asset], 'price', 500, '1d')
    ma_short = talib.SMA(prices[context.asset].values, timeperiod=context.MA_SHORT)
    ma_long = talib.SMA(prices[context.asset].values, timeperiod=context.MA_LONG)
    rsi = talib.RSI(prices[context.asset].values, timeperiod=context.RSI_PERIOD)

    is_bear_market = ma_short[-1] < ma_long[-1]
    is_bull_market = not is_bear_market

    # shift thresholds
    rsi_top = context.RSI_TOP + (context.RSI_SHIFT if is_bull_market else 0)
    rsi_bottom = context.RSI_BOTTOM - (context.RSI_SHIFT if is_bear_market else 0)

    window = context.RSI_PERIOD * context.PROB_WINDOW
    probability_bottom = probability(
        rsi[-window:], thresholds=(context.RSI_BOTTOM, context.RSI_TOP), what='bottom')

    #is_bull_market = True
    #probability_bottom = 50

    # Trading logic # is_bull_market and probability_bottom >= context.MIN_PROB
    if is_bull_market:
        if rsi[-2] > rsi_bottom and rsi[-1] < rsi_bottom:
            # buy
            order_target_percent(context.asset, 1.)
            context.days_ago = 0
            #print('buy', prices[context.asset].values[-2:], rsi[-1], rsi_bottom)
        elif rsi[-2] < rsi_top and rsi[-1] > rsi_top:
            order_target_percent(context.asset, 0)
    elif is_bear_market:
        order_target_percent(context.asset, 0)

    open_total = np.sum([x.amount * x.cost_basis
                         for x in context.portfolio.positions.values()])
    if open_total:
        context.days_ago += 1

    if context.days_ago > context.MAX_LONG and not get_open_orders():
        order_target_percent(context.asset, 0)
        context.days_ago = 0

    # Save values for later inspection
    record(open_total=open_total,
           probability_bottom=probability_bottom
          )


def rle(inarray):
    """
    Run length encoding. Partial credit to R rle function.
    Multi datatype arrays catered for including non Numpy

    Wiki: http://bit.ly/wiki_rle
    Source: htts://stackoverflow.com

    returns:
        tuple (runlengths, startpositions, values)
    """
    ia = np.array(inarray)                  # force numpy
    n = len(ia)
    if n == 0:
        return (None, None, None)
    else:
        y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions
        return(z, p, ia[i])


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
