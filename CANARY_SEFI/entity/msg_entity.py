import json


class MsgEntity(object):
    def __init__(self, status, code, msg):
        self.status = status
        self.code = code
        self.msg = msg

    def msg2json(std):
        return json.dumps({
            'status': std.status,
            'code': std.code,
            'msg': std.msg
        })
