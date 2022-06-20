from flask_socketio import SocketIO, join_room, send, leave_room, emit

name_space = '/websocket'


def get_socket(app):
    socketio = SocketIO()
    socketio.init_app(app, cors_allowed_origins='*', async_mode='threading')

    @socketio.on('connect', namespace='/websocket')
    def connected_msg():
        print('client connected!')

    @socketio.on('disconnect', namespace='/websocket')
    def disconnect_msg():
        print('client disconnected!')

    @socketio.on('join', namespace='/websocket')
    def on_join(data):
        username = data['username']
        room = data['room']
        join_room(room)
        emit("join_room", room, room=room)

    @socketio.on('leave', namespace='/websocket')
    def on_leave(data):
        username = data['username']
        room = data['room']
        leave_room(room)
        send(username + ' has left the room.', room=room)

    return socketio
