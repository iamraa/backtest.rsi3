import pytz
import pandas as pd
from pandas_datareader.data import DataReader
from collections import OrderedDict


def load(symbols, start, end):
    data = OrderedDict()
    for symbol in symbols:
        data[symbol] = DataReader(symbol, data_source='google', start=start,
                                  end=end)

    panel = pd.Panel(data)

    panel.minor_axis = ['open', 'high', 'low', 'close', 'volume']
    panel.major_axis = panel.major_axis.tz_localize(pytz.utc)

    return panel
