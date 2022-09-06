import random
import string


class BatchFlag(object):
    def __init__(self):
        self.batch_id = None
        self.new_batch()

    def new_batch(self):
        self.batch_id = ''.join(random.sample(string.ascii_letters + string.digits, 8))

    def set_batch(self, batch_id):
        self.batch_id = batch_id


batch_flag = BatchFlag()
