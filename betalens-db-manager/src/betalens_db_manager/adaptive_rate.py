"""Shared subscription/endpoint feedback; credentials never leave the digest."""
import time
from .util import digest


def scope(source,endpoint=''):
    return 'v2:'+digest(source.settings.token)[:20]+(':'+endpoint if endpoint else '')


def initialize(state):
    with state.connect() as c:c.execute('CREATE TABLE IF NOT EXISTS adaptive_rate(scope TEXT PRIMARY KEY,rpm REAL NOT NULL,ceiling REAL NOT NULL,last_limit REAL NOT NULL,last_raise REAL NOT NULL)')


def effective(source,endpoint,ceiling):
    initialize(source.state);key=scope(source,endpoint);instant=time.time()
    with source.state.connect() as c:
        c.execute('BEGIN IMMEDIATE');r=c.execute('SELECT * FROM adaptive_rate WHERE scope=?',(key,)).fetchone()
        if r is None:return ceiling
        rpm=min(float(r['rpm']),ceiling)
        if instant-r['last_limit']>=300 and instant-r['last_raise']>=60:
            rpm=min(ceiling,rpm+ceiling*.1);c.execute('UPDATE adaptive_rate SET rpm=?,last_raise=? WHERE scope=?',(rpm,instant,key))
        return rpm


def cool(source,endpoint,seconds,ceiling,account=False):
    initialize(source.state);key=scope(source,'' if account else endpoint);instant=time.time()
    with source.state.connect() as c:
        c.execute('BEGIN IMMEDIATE');old=c.execute('SELECT rpm FROM adaptive_rate WHERE scope=?',(key,)).fetchone()
        rate=max(10,min(ceiling,old[0] if old else ceiling)/2)
        c.execute('INSERT OR REPLACE INTO adaptive_rate VALUES(?,?,?,?,?)',(key,rate,ceiling,instant,instant))
        old=c.execute('SELECT next FROM request_rate WHERE scope=?',(key,)).fetchone()
        c.execute('INSERT OR REPLACE INTO request_rate VALUES(?,?)',(key,max(instant+seconds,old[0] if old else 0)))
