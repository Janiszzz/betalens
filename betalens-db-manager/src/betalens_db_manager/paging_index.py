"""Exact per-attempt page fingerprints with bounded SQLite cache.

This is disposable derived state, not a recovery journal. Resume reconstructs
it from SHA-verified source archives. It never grants completeness on its own.
"""
import sqlite3
import tempfile
import threading
from .util import digest


class PageIndex:
    def __init__(self):
        self.directory = tempfile.TemporaryDirectory(prefix='betalens-page-index-')
        self.connection = sqlite3.connect(self.directory.name+'/index.sqlite3', check_same_thread=False)
        self.connection.executescript('''
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA cache_size=-8192;
            CREATE TABLE fingerprints(chain TEXT,hash TEXT,page INTEGER,
                PRIMARY KEY(chain,hash)) WITHOUT ROWID;
        ''')
        self.lock = threading.RLock()

    @staticmethod
    def chain(request, page):
        omitted = {page.get('offset_param', 'offset'), page.get('limit_param', 'limit')}
        return digest([request['source_id'], request['endpoint'], bool(request.get('dependency_only')),
                       {k:v for k,v in request['parameters'].items() if k not in omitted}])

    def add(self, chain, offset, data, check=False):
        hashes = {digest(dict(zip(data['fields'], row))) for row in data['items']}
        if check and len(hashes) != len(data['items']):
            return True
        with self.lock, self.connection:
            if check:
                # SQLite variable limits are avoided by fixed-size packets.
                values = list(hashes)
                for start in range(0, len(values), 500):
                    part = values[start:start+500]
                    if self.connection.execute('SELECT 1 FROM fingerprints WHERE chain=? AND page<? AND hash IN ('+
                            ','.join('?' for _ in part)+') LIMIT 1', (chain, offset, *part)).fetchone():
                        return True
            self.connection.executemany('INSERT INTO fingerprints VALUES(?,?,?) ON CONFLICT(chain,hash) '
                'DO UPDATE SET page=min(page,excluded.page)', ((chain,h,offset) for h in hashes))
        return False

    def close(self):
        with self.lock:
            self.connection.close()
            self.directory.cleanup()
