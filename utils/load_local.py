import pytz
import pandas as pd
from utils.db import Db, get_prices

# connect to Db
_ = Db(host="localhost", user="developer", password="1", db="go_finance")

def load(symbols, start, end, is_adj=True):
    data = dict()

    # загружаем цены
    r = get_prices(symbols=symbols, dt_from=end, period=(end - start).days, is_adj=is_adj)

    for symbol in symbols:
        symbol_data = r['symbol'] == symbol
        data[symbol] = pd.DataFrame({
            'open': r['open'][symbol_data],
            'high': r['high'][symbol_data],
            'low': r['low'][symbol_data],
            'close': r['close'][symbol_data],
            'volume': r['volume'][symbol_data],
        }, index=r['dt'][symbol_data])

    panel = pd.Panel(data)

    panel.major_axis = panel.major_axis.tz_localize(pytz.utc)

    return panel
