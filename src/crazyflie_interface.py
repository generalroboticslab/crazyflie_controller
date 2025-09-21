from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
import numpy as np
import cflib.crtp
import time
import threading

class CrazyflieInterface:
    def __init__(self, uri="radio://0/70/2M/E7E7E7E701"):
        self.euler = None
        self.rate_readings = None
        self.battery_voltage = None
        self.lock = threading.Lock()
        # Initialize the low-level drivers
        cflib.crtp.init_drivers()
        self.cf = Crazyflie(rw_cache='./cache')
        self.scf = SyncCrazyflie(uri, cf=self.cf)
        self.scf.open_link()
        
        # Arm the Crazyflie
        self.cf.platform.send_arming_request(True)
        time.sleep(1.0)

        logconf = LogConfig(name='Euler', period_in_ms=10) # frequency of 100Hz
        logconf.add_variable('stateEstimate.roll', 'float')
        logconf.add_variable('stateEstimate.pitch', 'float') # remember this is flipped
        logconf.add_variable('stateEstimate.yaw', 'float')
        logconf.add_variable('gyro.x', 'float')
        logconf.add_variable('gyro.y', 'float')
        logconf.add_variable('gyro.z', 'float')
        logconf.add_variable('pm.vbat', 'FP16')
        self.cf.log.add_config(logconf)
        logconf.data_received_cb.add_callback(self.euler_cb)
        logconf.start()

    def euler_cb(self, _, data, __):
        with self.lock:
            self.euler = np.array([data['stateEstimate.roll'], # degrees
                        -data['stateEstimate.pitch'],
                        data['stateEstimate.yaw']])
            self.rate_readings = np.array([data['gyro.x'], # degrees/s
                                -data['gyro.y'],
                                data['gyro.z']])
            self.battery_voltage = data['pm.vbat']

    def send_control(self, omega, thrust):
        omega[1] = -omega[1]  # Invert the pitch rate to match the Crazyflie legacy mode
        self.cf.commander.send_manual_setpoint(*omega, thrust)

    def set_led_blue(self):
        """Set all LEDs to solid blue"""
        self.cf.param.set_value('ring.effect', '7')  # Solid color effect
        self.cf.param.set_value('ring.solidRed', '0')
        self.cf.param.set_value('ring.solidGreen', '0')
        self.cf.param.set_value('ring.solidBlue', '255')

    def set_led_orange(self):
        """Set all LEDs to solid orange"""
        self.cf.param.set_value('ring.effect', '7')  # Solid color effect
        self.cf.param.set_value('ring.solidRed', '255')
        self.cf.param.set_value('ring.solidGreen', '165')
        self.cf.param.set_value('ring.solidBlue', '0')

    def set_led_red(self):
        """Set all LEDs to solid red"""
        self.cf.param.set_value('ring.effect', '7')  # Solid color effect
        self.cf.param.set_value('ring.solidRed', '255')
        self.cf.param.set_value('ring.solidGreen', '0')
        self.cf.param.set_value('ring.solidBlue', '0')

    def set_led_off(self):
        """Turn off all LEDs"""
        self.cf.param.set_value('ring.effect', '0')  # Off effect

    def stop(self):
        for _ in range(30):
            self.cf.commander.send_setpoint(0, 0, 0, 0)  # Stop the motors
            time.sleep(0.1)
        self.scf.close_link()

    def get_euler_degs(self):
        with self.lock:
            return self.euler
    
    def get_omega_degs(self):
        with self.lock:
            return self.rate_readings
    
    def get_battery_voltage(self):
        with self.lock:
            return self.battery_voltage

if __name__ == "__main__":
    cf_interface = CrazyflieInterface()
    # Unlock startup thrust protection
    cf_interface.cf.commander.send_setpoint(0.0, 0.0, 0.0, 0)  # Example control command
    
    # Example LED usage
    print("Setting LED to blue...")
    cf_interface.set_led_blue()
    time.sleep(2)
    
    print("Setting LED to orange...")
    cf_interface.set_led_orange()
    time.sleep(2)
    
    try:
        while True:
            euler_angles = cf_interface.get_euler_degs()
            rate_readings = cf_interface.get_omega_degs()
            battery_voltage = cf_interface.get_battery_voltage()
            cf_interface.send_control([0.0, 20.0, 10.0], 35000)  # Example control command
            time.sleep(0.02)
            if euler_angles is not None:
                print(f"Euler rates: Roll: {rate_readings[0]:.2f}, Pitch: {rate_readings[1]:.2f}, Yaw: {rate_readings[2]:.2f}, Battery: {battery_voltage}V")
                pass
            else:
                print("Waiting for Euler angles...")
    except KeyboardInterrupt:
        print("Stopped Crazyflie interface.")
        cf_interface.set_led_off()  # Turn off LEDs before stopping
        cf_interface.stop()