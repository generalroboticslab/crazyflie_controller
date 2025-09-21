import threading
import queue
import csv

class Logger:
    def __init__(self, filename="/home/generalroboticslab/Desktop/crazyflie-controller/log.csv"):
        self.queue = queue.Queue()
        self.filename = filename
        self.running = True
        self.thread = threading.Thread(target=self._logger_thread)
        self.thread.start()

    def log(self, timestamp, obs_input, angvel, thrust, desired_obs):
        row = [timestamp] + list(obs_input) + list(angvel) + [thrust] + list(desired_obs)
        self.queue.put(row)

    def _logger_thread(self):
        with open(self.filename, "w", newline='') as f:
            writer = csv.writer(f)
            # Detailed header
            header = (
                ["timestamp"] +
                ["pos_x", "pos_y", "pos_z"] +
                ["quat_w", "quat_x", "quat_y", "quat_z"] +
                ["vel_x", "vel_y", "vel_z"] +
                ["omega_x", "omega_y", "omega_z"] +
                ["angvel_x", "angvel_y", "angvel_z"] +
                ["thrust"] + 
                ["desired_pos_x", "desired_pos_y", "desired_pos_z"] +
                ["desired_quat_w", "desired_quat_x", "desired_quat_y", "desired_quat_z"] +
                ["desired_vel_x", "desired_vel_y", "desired_vel_z"] +
                ["desired_omega_x", "desired_omega_y", "desired_omega_z"]
            )
            writer.writerow(header)

            while self.running or not self.queue.empty():
                try:
                    item = self.queue.get(timeout=0.1)
                    writer.writerow(item)
                except queue.Empty:
                    continue

    def close(self):
        self.running = False
        self.thread.join()
