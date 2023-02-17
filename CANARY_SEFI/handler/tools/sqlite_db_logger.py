class SqliteDBLogger:
    def __init__(self, conn):
        self.conn = conn

        def dict_factory(cursor, row):
            data = {}
            for idx, col in enumerate(cursor.description):
                data[col[0]] = row[idx]
            return data
        self.conn.row_factory = dict_factory

        self.debug_log = True

    def insert_log(self, sql, args):
        cursor = self.conn.cursor()
        cursor.execute(sql, args)
        log_id = int(cursor.lastrowid)
        cursor.close()
        self.conn.commit()
        return log_id

    def query_logs(self, sql, args):
        cursor = self.conn.cursor()
        cursor.execute(sql, args)
        values = cursor.fetchall()
        cursor.close()
        return values

    def query_log(self, sql, args):
        cursor = self.conn.cursor()
        cursor.execute(sql, args)
        values = cursor.fetchone()
        cursor.close()
        return values

    def update_log(self, sql, args):
        cursor = self.conn.cursor()
        cursor.execute(sql, args)
        cursor.close()
        self.conn.commit()

    def finish(self):
        self.conn.commit()
