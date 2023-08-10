import random

from colorama import Fore, Style
from flask_socketio import SocketIO, join_room, leave_room, emit
from tqdm import tqdm

from canary_sefi.task_manager import task_manager


class RealtimeReport:

    def __init__(self):
        self.rooms = []

    def console_log(self, msg, fore, type="INFO", save_db=True, send_msg=True, show_task=False,
                    show_step_sequence=False):
        if save_db:
            task_manager.sys_log_logger.new_console_record(msg, type)
        # 处理额外信息附加
        if show_task:
            msg = "[ TASK {} ] ".format(task_manager.task_token) + msg
        if show_step_sequence:
            msg = "[ STEP {} ] ".format(task_manager.sys_log_logger.step_sequence) + msg

        if len(self.rooms) != 0 and send_msg:
            self.send_realtime_msg(msg, type)
        tqdm.write(fore + msg + Fore.RESET)

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
            room_tmp = str(random.randint(10000, 100000))
            self.rooms.append(room_tmp)
            join_room(room_tmp)
            emit("join_room", room_tmp, room=room_tmp)

        @socketio.on('leave_room', namespace='/realtime_msg')
        def on_leave(room):
            self.rooms.remove(room)
            leave_room(room)
            emit("disconnect", None, room=room)

        return socketio

    def send_realtime_msg(self, msg, type=None):
        info = {
            "type": type,
            "msg": msg
        }
        for room in self.rooms:
            emit("message", info, room=room, namespace='/realtime_msg')

    def send_disconnect(self):
        for room in self.rooms:
            emit("disconnect", None, room=room, namespace='/realtime_msg')
        self.rooms = []


reporter = RealtimeReport()
