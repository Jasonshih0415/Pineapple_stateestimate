try:
    import cv2
except ImportError:
    raise ImportError("Please install opencv-python: pip install opencv-python")

import numpy as np
import threading
import time

class GUITeleop:
    def __init__(self, config_init=[0,0,0], lin_step=0.2, ang_step=0.2, max_lin=1.0, max_ang=3.0,
                 height_init=0.3, height_step=0.01, min_height=0.2, max_height=0.35):
        self.cmd_vel = np.array(config_init, dtype=np.float32)
        self.cmd_height = height_init
        
        self.lin_step = lin_step
        self.ang_step = ang_step
        self.height_step = height_step
        
        self.max_lin = max_lin
        self.max_ang = max_ang
        self.min_height = min_height
        self.max_height = max_height
        
        self.window_name = "Control Panel"
        
        # Colors (B, G, R)
        self.bg_color = (30, 30, 30)
        self.text_color = (220, 220, 220)
        self.highlight_color = (255, 255, 255)
        self.bar_bg_color = (60, 60, 60)
        self.lin_bar_color = (235, 137, 52)   # Mujoco-like blue/cyan accent
        self.ang_bar_color = (52, 137, 235)   # Orange accent
        self.height_bar_color = (50, 205, 50) # Green accent
        
        # Dimensions
        self.width = 500
        self.height = 360  # Increased height to fit extra bar
        self.margin = 30
        
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def get_command(self):
        with self.lock:
            return self.cmd_vel.copy()
            
    def get_height_command(self):
        with self.lock:
            return self.cmd_height

    def _run(self):
        # Window must be created and updated in the same thread
        while self.running:
            self._update_display()
            key = cv2.waitKey(30) & 0xFF  # ~30Hz update rate
            
            with self.lock:
                if key != 255: 
                    if key == ord('w'):
                        self.cmd_vel[0] = np.clip(self.cmd_vel[0] + self.lin_step, -self.max_lin, self.max_lin)
                    elif key == ord('s'):
                        self.cmd_vel[0] = np.clip(self.cmd_vel[0] - self.lin_step, -self.max_lin, self.max_lin)
                    elif key == ord('a'):
                        self.cmd_vel[2] = np.clip(self.cmd_vel[2] + self.ang_step, -self.max_ang, self.max_ang)
                    elif key == ord('d'):
                        self.cmd_vel[2] = np.clip(self.cmd_vel[2] - self.ang_step, -self.max_ang, self.max_ang)
                    elif key == ord('i'):
                        self.cmd_height = np.clip(self.cmd_height + self.height_step, self.min_height, self.max_height)
                    elif key == ord('k'):
                        self.cmd_height = np.clip(self.cmd_height - self.height_step, self.min_height, self.max_height)
                    elif key == ord(' '):
                        self.cmd_vel[:] = 0.0
                    elif key == 27: # ESC
                        self.cmd_vel[:] = 0.0
        
        cv2.destroyWindow(self.window_name)

    def _draw_bar(self, img, value, max_val, y_pos, label, color):
        # Bar background
        bar_width = self.width - 2 * self.margin
        bar_height = 16
        bar_x = self.margin
        center_x = bar_x + bar_width // 2
        
        cv2.rectangle(img, (bar_x, y_pos), (bar_x + bar_width, y_pos + bar_height), self.bar_bg_color, -1)
        
        # Zero mark
        cv2.line(img, (center_x, y_pos - 2), (center_x, y_pos + bar_height + 2), (150, 150, 150), 1)
        
        # Value bar
        norm_val = value / max_val
        fill_width = int(norm_val * (bar_width / 2))
        
        if fill_width > 0:
            cv2.rectangle(img, (center_x, y_pos), (center_x + fill_width, y_pos + bar_height), color, -1)
        else:
            cv2.rectangle(img, (center_x + fill_width, y_pos), (center_x, y_pos + bar_height), color, -1)
            
        # Label and value text
        cv2.putText(img, label, (bar_x, y_pos - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1, cv2.LINE_AA)
        val_text = f"{value:.2f}"
        (w, h), _ = cv2.getTextSize(val_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(img, val_text, (bar_x + bar_width - w, y_pos - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.highlight_color, 1, cv2.LINE_AA)

    def _update_display(self):
        # Create dark background
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = self.bg_color
        
        # Snapshot current values for display
        with self.lock:
             current_vel = self.cmd_vel.copy()
             current_height = self.cmd_height
        
        # Header
        cv2.putText(img, "TELEOP CONTROL", (self.margin, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, self.highlight_color, 1, cv2.LINE_AA)
        cv2.line(img, (self.margin, 65), (self.width - self.margin, 65), (100, 100, 100), 1)
        
        # Draw Bars
        self._draw_bar(img, current_vel[0], self.max_lin, 120, "Linear Velocity (x)", self.lin_bar_color)
        self._draw_bar(img, current_vel[2], self.max_ang, 190, "Angular Velocity (yaw)", self.ang_bar_color)
        
        # Height Bar (Custom drawing because it's strictly positive range)
        self._draw_height_bar(img, current_height, self.min_height, self.max_height, 260, "Robot Height", self.height_bar_color)

        # Footer Help
        help_text = "SPACE: STOP | ESC: RESET | I/K: HEIGHT"
        help_y = self.height - 20
        (w, h), _ = cv2.getTextSize(help_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(img, help_text, ((self.width - w)//2, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
        
        cv2.imshow(self.window_name, img)

    def _draw_height_bar(self, img, value, min_val, max_val, y_pos, label, color):
        # Bar background
        bar_width = self.width - 2 * self.margin
        bar_height = 16
        bar_x = self.margin
        
        cv2.rectangle(img, (bar_x, y_pos), (bar_x + bar_width, y_pos + bar_height), self.bar_bg_color, -1)
        
        # Calculate fill based on range [min_val, max_val]
        # Avoid division by zero
        range_val = max_val - min_val
        if range_val <= 0: range_val = 1.0
        
        ratio = (value - min_val) / range_val
        ratio = np.clip(ratio, 0.0, 1.0)
        
        fill_width = int(ratio * bar_width)
        
        cv2.rectangle(img, (bar_x, y_pos), (bar_x + fill_width, y_pos + bar_height), color, -1)
            
        # Label and value text
        cv2.putText(img, label, (bar_x, y_pos - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1, cv2.LINE_AA)
        val_text = f"{value:.3f}"
        (w, h), _ = cv2.getTextSize(val_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(img, val_text, (bar_x + bar_width - w, y_pos - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.highlight_color, 1, cv2.LINE_AA)


    def close(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
