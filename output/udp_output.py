import socket
import struct


class OpenTrackUDP:
    def __init__(self, host="127.0.0.1", port=4242):
        self.address = (host, port)
        self.socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM
        )

    def send_pose(self, x, y, z, yaw, pitch, roll):

        packet = struct.pack(
            "<6d",
            x,
            y,
            z,
            yaw,
            pitch,
            roll
        )

        self.socket.sendto(
            packet,
            self.address
        )

    def close(self):
        self.socket.close()