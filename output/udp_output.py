import socket
import struct


class OpenTrackUDP:

    def __init__(self, host="127.0.0.1", port=4242):
        # Save the destination address.
        #
        # 127.0.0.1 means "this computer", so the data does not
        # leave the machine. OpenTrack will be listening locally
        # on this port for incoming head-tracking data.
        self.address = (host, port)

        # Create a UDP socket.
        #
        # AF_INET  = use IPv4 addresses
        # SOCK_DGRAM = use UDP instead of TCP
        #
        # UDP is useful here because we are sending small,
        # real-time pose updates many times per second.
        # If one packet is lost, the next frame will replace it anyway.
        self.socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM
        )

    def send_pose(self, x, y, z, yaw, pitch, roll):
        # Pack the six pose values into a binary packet
        # in the format OpenTrack expects.
        #
        # "<6d" means:
        # <  = little-endian byte order
        # 6  = six values
        # d  = each value is a double-precision floating-point number
        #
        # Packet order:
        # x, y, z, yaw, pitch, roll
        packet = struct.pack(
            "<6d",
            x,
            y,
            z,
            yaw,
            pitch,
            roll
        )

        # Send the packed pose data to OpenTrack.
        #
        # self.address will normally be:
        # ("127.0.0.1", 4242)
        self.socket.sendto(
            packet,
            self.address
        )

    def close(self):
        # Close the UDP socket when the tracker shuts down
        # so the operating system can release the resource.
        self.socket.close()