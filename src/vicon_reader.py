import rospy
from geometry_msgs.msg import TransformStamped
import threading
import numpy as np

class ViconReader:
    def __init__(self, topic="/vicon/cf1/cf1"):
        self.lock = threading.Lock()
        self.pos = None
        self.vel = None
        self.prev_pos = None
        self.prev_time = None

        rospy.Subscriber(topic, TransformStamped, self.callback)

    def callback(self, msg):
        x, y, z = msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z
        t = msg.header.stamp.to_sec()

        with self.lock:
            if self.prev_time is None:
                self.prev_time, self.prev_pos = t, (x, y, z)
                self.pos = (x, y, z)
                self.vel = (0, 0, 0)
                return
            dt = t - self.prev_time
            if dt > 0:
                self.vel = tuple((np.array([x, y, z]) - np.array(self.prev_pos)) / dt)
                self.pos = (x, y, z)
                self.prev_pos = (x, y, z)
                self.prev_time = t

    def get_state(self):
        with self.lock:
            return self.pos, self.vel

if __name__ == "__main__":
    rospy.init_node("vicon_reader_node")
    vicon_reader = ViconReader()
    rate = rospy.Rate(50)  # 50 Hz

    while not rospy.is_shutdown():
        pos, vel = vicon_reader.get_state()
        print(f"Position: {pos}, Velocity: {vel}")
        rate.sleep()