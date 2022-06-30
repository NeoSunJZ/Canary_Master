import random
import string


class BatchFlag(object):
    def __init__(self):
        self.batch_id = None
        self.new_batch()

    def new_batch(self):
        self.batch_id = ''.join(random.sample(string.ascii_letters + string.digits, 8))


batch_flag = BatchFlag()
