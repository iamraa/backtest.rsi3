import psycopg2
import psycopg2.extras
from datetime import datetime, date, timedelta
import numpy as np


class Db(object):
    """
    Класс работы с базой данных.
    """
    _instance = None
    _connection = None

    def __new__(self, **kwargs):
        if Db._instance is None:
            Db._instance = object.__new__(self)
            print('create Db class...')
        return Db._instance

    def __init__(self, host="localhost", user=None, password=None, db=None):
        if Db._instance is not None and user is not None:
            print('connect to {}...'.format(db))
            try:
                self._connection = psycopg2.connect(
                    "host='{0}' user='{1}' password='{2}' dbname='{3}'".format(
                        host, user, password, db))
            except psycopg2.Error as err:
                print("Connection error: {}".format(err))
                self._connection.close()

            # включаем float вместо Decimal()
            dec2float = psycopg2.extensions.new_type(
                psycopg2.extensions.DECIMAL.values,
                'dec2float',
                lambda value, curs: float(value) if value is not None else None)
            psycopg2.extensions.register_type(dec2float)

    def query(self, sql, params=None, cursor='list'):
        if not self._connection:
            return False

        data = False

        if cursor == 'dict':
            # Assoc cursor
            factory = psycopg2.extras.DictCursor
        else:
            # Standard cursor
            factory = psycopg2.extensions.cursor
        with self._connection.cursor(cursor_factory=factory) as cur:
            try:
                cur.execute(sql, params)
                data = cur.fetchall()
            except psycopg2.Error as err:
                self._connection.rollback()
                print("Query error: {0}\n{1}".format(
                    err,
                    cur.query.decode('utf-8') if cur.query else 'None'))

        return data

    def query_exec(self, sql, params=None, is_many=False, is_return=False):
        if not self._connection:
            return False

        with self._connection.cursor() as cur:
            try:
                if is_many:
                    cur.executemany(sql, params)
                else:
                    cur.execute(sql, params)
                self._connection.commit()

                # fetch response
                if is_return:
                    r = cur.fetchone()
                    return cur.rowcount, r

                # return affected rows
                return cur.rowcount

            except psycopg2.Error as err:
                self._connection.rollback()
                print("Query error: {0}\n{1}".format(
                    err,
                    cur.query.decode('utf-8') if cur.query else 'None'))

        return False


def get_prices(symbols=None, dt_from=date.today(), period=600, is_adj=True):
    assert symbols  # empty symbols

    dt_to = dt_from - timedelta(days=period)
    cond = {
        'symbol': symbols,
        'to': dt_from,
        'from': dt_to
    }

    sql = """
        SELECT
            string_agg(symbol::text, ',') AS symbol_list
            , string_agg(dt::text, ',') AS dt_list
            , string_agg(open::text, ',') AS open_list
            , string_agg(high::text, ',') AS high_list
            , string_agg(low::text, ',') AS low_list
            , string_agg("close"::text, ',') AS close_list
            , string_agg(volume::text, ',') AS volume_list
            , string_agg(adj::text, ',') AS adj_list
        FROM v_prices_fast
        WHERE symbol IN ('{0}')
        AND dt BETWEEN %(from)s AND %(to)s
        """.format("','".join([str(i) for i in cond['symbol']]))

    del cond['symbol']  # Remove from conditions
    r = Db().query(sql, cond)[0]

    # print(sql, cond)

    d = {
        'symbol': np.array(r[0].split(',')),
        'dt': np.array(r[1].split(','), dtype='datetime64'),
        'open': np.fromstring(r[2], sep=',') / 10000,
        'high': np.fromstring(r[3], sep=',') / 10000,
        'low': np.fromstring(r[4], sep=',') / 10000,
        'close': np.fromstring(r[5], sep=',') / 10000,
        'volume': np.fromstring(r[6], sep=','),
    }
    if is_adj:
        print('adjusted')
        adj = np.fromstring(r[7], sep=',') / 10000
        adj_exists = adj > 0
        ratio = adj[adj_exists] / d['close'][adj_exists]

        # print(len(d['close']), len(adj), len(adj_exists), len(ratio))
        # np.around(arr['low'], decimals=4)

        d['open'][adj_exists] = d['open'][adj_exists] * ratio
        d['high'][adj_exists] = d['high'][adj_exists] * ratio
        d['low'][adj_exists] = d['low'][adj_exists] * ratio
        d['close'][adj_exists] = d['close'][adj_exists] * ratio

    return d
