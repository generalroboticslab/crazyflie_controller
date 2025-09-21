"""
Very roughly tuned PD stabilizer. 
Used only to keep the drone steady enough for takeoff and landing 
before the main MPPI controller takes over.
"""
import numpy as np

class AttitudeStabilizer:
    def __init__(self):
        self.Kp = np.array([10.0, 10.0, 13.75])  # [roll, pitch, yaw] gains
        self.Kd = np.array([-0.2, 0.2, 4.0])  # [roll, pitch, yaw] gains
        self.Kp_z = 3.2
        self.Kd_z = 1.0
        self.gravity_magnitude = 9.81
        self.gravity_vector = np.array([0, 0, -self.gravity_magnitude])
        self.mass = 0.041 #0.035  # mass in kg

    def control(self, euler_deg, euler_rate, pos, vel):
        # Input: (roll, pitch, yaw) in degrees
        desired_pos = np.array([0, 0, 0.6])  # Desired position
        pos_error_xy = desired_pos[:2] - pos[:2]
        pos_Kp = np.array([5.5, 5.5])
        pitch_correction = -pos_Kp[0] * pos_error_xy[0]
        roll_correction  = -pos_Kp[1] * pos_error_xy[1]
        
        target_euler = np.array([roll_correction, pitch_correction, 0.0])
        euler_rate = np.array(euler_rate)
        euler_rate[1] = -euler_rate[1]
        euler_deg[1] = -euler_deg[1]
        
        angle_error = np.array(euler_deg) - target_euler
        omega_des = -self.Kp * angle_error - self.Kd * euler_rate
        omega_des[1] = -omega_des[1]
        
        # --- Altitude control (PD) ---
        z_des = np.minimum(0.6, pos[2] + 0.05)
        z_error = pos[2] - z_des # -0.5
        z_vel_error = vel[2] 
        acc_des_z = -self.Kp_z * z_error - self.Kd_z * z_vel_error - self.gravity_vector[2]
        
        return omega_des, self.acc_to_pwm(acc_des_z)
    
    def control_landing(self, euler_deg, euler_rate, pos, vel):
        # Input: (roll, pitch, yaw) in degrees
        desired_pos = np.array([0, 0, 0.07])  # Desired position
        pos_error_xy = desired_pos[:2] - pos[:2]
        pos_Kp = np.array([5.5, 5.5])
        pitch_correction = -pos_Kp[0] * pos_error_xy[0]
        roll_correction  = -pos_Kp[1] * pos_error_xy[1]
        
        target_euler = np.array([roll_correction, pitch_correction, 0.0])
        euler_rate = np.array(euler_rate)
        euler_rate[1] = -euler_rate[1]
        euler_deg[1] = -euler_deg[1]
        
        angle_error = np.array(euler_deg) - target_euler
        omega_des = -self.Kp * angle_error - self.Kd * euler_rate
        omega_des[1] = -omega_des[1]
        
        # --- Altitude control (PD) ---
        z_des = pos[2] - 0.5
        z_error = pos[2] - z_des # -0.5
        z_vel_error = vel[2] 
        acc_des_z = -self.Kp_z * z_error - self.gravity_vector[2]
        # - self.Kd_z * z_vel_error - self.gravity_vector[2]
        
        return omega_des, self.acc_to_pwm(acc_des_z)
        
    
    def acc_to_pwm(self, acc_des_z):
        pwm =  int((acc_des_z) * 132000 * self.mass)
        pwm = np.clip(pwm, 10000, 60000)
        return pwm
    
if __name__ == "__main__":
    hover_controller = AttitudeStabilizer()
    
    print(hover_controller.thrust_to_pwm(36/4))