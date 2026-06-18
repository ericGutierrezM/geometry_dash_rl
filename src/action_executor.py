from pynput.keyboard import Key, Controller as KeyboardController
import time
import pyautogui

class ActionExecutor:
    def __init__(self, monitor):
        self.keyboard = KeyboardController()
        self.monitor = monitor
        # PyAutoGUI failsafe disable (prevents crashes if you move your physical mouse)
        pyautogui.FAILSAFE = False

    def restart_level(self):
        time.sleep(1) # Wait for death animation
        
        # Move to the button coordinates
        # Double check these! If your window moved, the click will miss.
        target_x, target_y = 422, 662 
                
        # Click 1: Focus the Geometry Dash window
        pyautogui.click(x=target_x, y=target_y)
        time.sleep(0.1) # Micro-pause for macOS to switch focus
        
        # Click 2: Actually press the restart button
        pyautogui.mouseDown(x=target_x, y=target_y, button='left')
        time.sleep(0.1) # Hold for 100ms so the game registers it
        pyautogui.mouseUp(x=target_x, y=target_y, button='left')
        
        time.sleep(0.5) # Wait for level to load

    from pynput.keyboard import Key, Controller as KeyboardController
import time
import pyautogui

class ActionExecutor:
    def __init__(self, monitor):
        self.keyboard = KeyboardController()
        self.monitor = monitor
        # PyAutoGUI failsafe disable (prevents crashes if you move your physical mouse)
        pyautogui.FAILSAFE = False

    def restart_level(self):
        time.sleep(1) # Wait for death animation
        
        # Move to the button coordinates
        # Double check these! If your window moved, the click will miss.
        target_x, target_y = 422, 662 
        
        print(f"CLICKANDO en {target_x}, {target_y}")
        
        # Click 1: Focus the Geometry Dash window
        pyautogui.click(x=target_x, y=target_y)
        time.sleep(0.1) # Micro-pause for macOS to switch focus
        
        # Click 2: Actually press the restart button
        pyautogui.mouseDown(x=target_x, y=target_y, button='left')
        time.sleep(0.1) # Hold for 100ms so the game registers it
        pyautogui.mouseUp(x=target_x, y=target_y, button='left')
        
        time.sleep(0.5) # Wait for level to load

    def act(self, action):
        if action == 1:
            # 1. Press the key
            self.keyboard.press(Key.space)
            
            # 2. Hold it for just 30 milliseconds (roughly 2 frames at 60FPS)
            # This is fast enough to avoid the "Sticky Key" double-jump, 
            # but long enough that the Geometry Dash engine won't miss the input.
            time.sleep(0.03) 
            
            # 3. Immediately release it!
            self.keyboard.release(Key.space)