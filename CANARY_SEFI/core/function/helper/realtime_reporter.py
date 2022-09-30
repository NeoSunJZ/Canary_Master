import random

from colorama import Fore, Style
from flask_socketio import SocketIO, join_room, emit
from tqdm import tqdm

from CANARY_SEFI.batch_manager import batch_flag
from CANARY_SEFI.core.function.helper.system_log import global_system_log


class RealtimeReport:

    def __init__(self):
        self.room = None

    def console_log(self, msg, fore, type="INFO", save_db=True, send_msg=True, show_batch=False,
                    show_step_sequence=False):
        if save_db:
            global_system_log.new_console_record(msg, type)
        # 处理额外信息附加
        if show_batch:
            msg = "[ BATCH {} ] ".format(batch_flag.batch_id) + msg
        if show_step_sequence:
            msg = "[ STEP {} ] ".format(global_system_log.step_sequence) + msg

        if self.room is not None and send_msg:
            self.send_realtime_msg(msg, type)
        tqdm.write(fore + msg)
        tqdm.write(Style.RESET_ALL)

    def get_socket(self, app):
        socketio = SocketIO()
        socketio.init_app(app, cors_allowed_origins='*', async_mode='threading')

        @socketio.on('connect', namespace='/realtime_msg')
        def connected_msg():
            self.console_log("[ WEB CONSOLE ] 连接已建立", Fore.RED, save_db=False, send_msg=False)

        @socketio.on('disconnect', namespace='/realtime_msg')
        def disconnect_msg():
            self.console_log("[ WEB CONSOLE ] 连接已断开", Fore.RED, save_db=False, send_msg=False)

        @socketio.on('join', namespace='/realtime_msg')
        def on_join():
            self.room = str(random.randint(10000, 100000))
            join_room(self.room)
            emit("join_room", self.room, room=self.room)

        return socketio

    def send_realtime_msg(self, msg, type=None):
        info = {
            "type": type,
            "msg": msg
        }
        emit("message", info, room=self.room, namespace='/realtime_msg')

    def send_disconnect(self):
        emit("disconnect", None, room=self.room, namespace='/realtime_msg')


reporter = RealtimeReport()
