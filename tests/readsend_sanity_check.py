from crazyflie_controller.src.crazyflie_interface import CrazyflieInterface
from crazyflie_controller.src.vicon_reader import ViconReader
import time
import rospy

cf_interface = CrazyflieInterface()
vicon_reader = ViconReader()
rospy.init_node("vicon_reader_node")
rate = 50  # Hz
time.sleep(1)  # Allow some time for initialization

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
    for _ in range(100):
        pos, vel = vicon_reader.get_state()
        euler_angles = cf_interface.get_euler_degs()
        rate_readings = cf_interface.get_omega_degs()
        battery_voltage = cf_interface.get_battery_voltage()
        cf_interface.send_control([0.0, 0.0, 0.0], 25000)  # Example control command
        time.sleep(0.02)
        print("=" * 40)
        print("Drone State")
        print("=" * 40)
        print(f"Position      ->  x: {pos[0]:.3f}, y: {pos[1]:.3f}, z: {pos[2]:.3f}")
        print(f"Velocity      ->  vx: {vel[0]:.3f}, vy: {vel[1]:.3f}, vz: {vel[2]:.3f}")
        print(f"Euler Angles  ->  Roll: {euler_angles[0]:.2f}, "
            f"Pitch: {euler_angles[1]:.2f}, Yaw: {euler_angles[2]:.2f}")
        print(f"Euler Rates   ->  Roll: {rate_readings[0]:.2f}, "
            f"Pitch: {rate_readings[1]:.2f}, Yaw: {rate_readings[2]:.2f}")
        print(f"Battery       ->  {battery_voltage:.2f} V")
        print("=" * 40)

    
    cf_interface.set_led_off()
    cf_interface.stop()
except KeyboardInterrupt:
    print("Stopped Crazyflie interface.")
    cf_interface.set_led_off()  # Turn off LEDs before stopping
    cf_interface.stop()