import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random

class GeometryDashSim(gym.Env):
    def __init__(self):
        super(GeometryDashSim, self).__init__()
        
        # --- 1. CONFIGURACIÓN DE LA PANTALLA VIRTUAL ---
        self.screen_width = 1000
        self.screen_height = 561
        self.max_distance = np.sqrt(self.screen_width ** 2 + self.screen_height ** 2)
        
        # El orden de clases exacto de tu FeatureExtractor
        self.CLASS_ORDER = ['block', 'coin', 'platform', 'player', 'portal', 'spaceship', 'spike']
        self.max_obstacles = 6
        
        # --- 2. ESPACIO DE ACCIÓN Y OBSERVACIÓN PERFECTO (84 elementos) ---
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(84,), dtype=np.float32)
        
        # --- 3. PARÁMETROS FÍSICOS EN PÍXELES ---
        self.player_x = 250          # Posición X fija del jugador en la pantalla
        self.ground_y = 420          # El suelo está abajo (píxel 550 de 720)
        self.gravity = 1.09375          # Gravedad (suma píxeles hacia abajo)
        self.jump_force = -18.046875     # Impulso de salto (resta píxeles hacia arriba)
        self.game_speed = 7.8125     # Velocidad a la que avanzan los pinchos (píxeles/frame)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Estado inicial del jugador (Coordenadas de pantalla: Y aumenta hacia abajo)
        self.player_y = float(self.ground_y)
        self.player_velocity_y = 0.0
        self.prev_player_y = float(self.ground_y)
        
        # Generar primer obstáculo (un pincho 'spike') a la derecha de la pantalla
        self.obstacle_x = float(self.screen_width - 100)
        self.obstacle_y = float(self.ground_y)
        self.obstacle_class = random.choice(['spike', 'block'])
        
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        reward = 0.1
        terminated = False

        self.prev_player_y = self.player_y
        
        # --- THE YOLO LAG SIMULATOR ---
        # 13 frames perfectly matches the ~0.14 distance jump seen in your YOLO logs!
        frames_to_skip = 3

        if action == 1 and (self.obstacle_x - self.player_x) > 320:
            reward -= 5.0
        
        # 1. El agente toma la decisión (El Impuesto Incondicional)
        if action == 1 and self.player_y >= self.ground_y:
            self.player_velocity_y = self.jump_force

        # 2. FAST-FORWARD: Simulamos el tiempo que tarda YOLO en pensar
        for _ in range(frames_to_skip):
            
            # Gravedad
            self.player_velocity_y += self.gravity
            self.player_y += self.player_velocity_y
            
            # Suelo
            if self.player_y >= self.ground_y:
                self.player_y = float(self.ground_y)
                self.player_velocity_y = 0.0
                
            # Movimiento del obstáculo
            self.obstacle_x -= self.game_speed
            dx = self.obstacle_x - self.player_x
            
            # Reset del obstáculo
            if dx < -50:
                self.obstacle_x = float(self.player_x + random.randint(500, 1500))
                self.obstacle_class = random.choice(['spike', 'block'])
                reward += 5.0
                dx = self.obstacle_x - self.player_x
                
            hit_x = False
            hit_y = False
            
            if self.obstacle_class == 'spike':
                hit_x = (-10 <= dx <= 25) 
                hit_y = (self.player_y >= (self.ground_y - 15))
            elif self.obstacle_class == 'block':
                hit_x = (-15 <= dx <= 35) 
                hit_y = (self.player_y >= (self.ground_y - 30))
                
            if hit_x and hit_y:
                terminated = True
                reward = -10.0  

            # --- THE FEAR ZONE (LA ZONA DE PÁNICO) ---
            # If the cube is on the ground, and the obstacle is dangerously close 
            # (under 160 pixels), we kill the agent. 
            # This mathematically forces the AI to jump when YOLO sees 0.22!
            if self.player_y >= self.ground_y and (0 < dx < 160):
                terminated = True
                reward = -10.0
                
        # Ajustamos el límite porque ahora cada paso avanza 13 veces más rápido
        truncated = self.steps > (3000 / frames_to_skip)
        
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        player_y_norm = self.player_y / self.screen_height
        velocity_y = self.player_y - self.prev_player_y
        player_velocity_y_norm = np.clip(velocity_y / self.screen_height, -1.0, 1.0)
        on_ground = 1.0 if self.player_y >= self.ground_y else 0.0
        
        ground_distance = self.screen_height - self.player_y
        ground_distance_norm = ground_distance / self.screen_height
        
        dx = max(0.0, self.obstacle_x - self.player_x)
        dy = self.obstacle_y - self.player_y
        distance = np.sqrt(dx**2 + dy**2)
        
        # --- EL FIX: RADARES INDEPENDIENTES ---
        # Ahora los sensores base imitan perfectamente a YOLO
        if self.obstacle_class == 'spike':
            next_spike_distance_norm = min(dx / self.screen_width, 1.0)
            next_platform_distance_norm = 1.0 # El radar de bloques no ve nada
        elif self.obstacle_class == 'block':
            next_spike_distance_norm = 1.0 # El radar de pinchos no ve nada
            next_platform_distance_norm = min(dx / self.screen_width, 1.0)
        else:
            next_spike_distance_norm = 1.0
            next_platform_distance_norm = 1.0
            
        # Construimos la base (6 elementos)
        state_values = [
            float(player_y_norm),
            float(player_velocity_y_norm),
            float(on_ground),
            float(ground_distance_norm),
            float(next_spike_distance_norm),
            float(next_platform_distance_norm)
        ]
        
        obstacle_features = []
        
        dx_norm = dx / self.screen_width
        dy_norm = np.clip(dy / self.screen_height, -1.0, 1.0)
        distance_norm = distance / self.max_distance
        collision_risk = 1.0 * max(1.0 - distance_norm, 0.0)
        
        # --- FIX 2c: DYNAMIC OBSERVATION ENCODING ---
        class_idx = self.CLASS_ORDER.index(self.obstacle_class)
        class_type_norm = float(class_idx) / (len(self.CLASS_ORDER) - 1)
        can_land_on_top = 1.0 if self.obstacle_class == 'block' else 0.0
        
        one_hot = [0.0] * len(self.CLASS_ORDER)
        one_hot[class_idx] = 1.0
        
        obstacle_features.append({
            "metrics": [dx_norm, dy_norm, distance_norm, class_type_norm, can_land_on_top, collision_risk],
            "one_hot": one_hot
        })
        
        while len(obstacle_features) < self.max_obstacles:
            obstacle_features.append({
                "metrics": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "one_hot": [0.0] * len(self.CLASS_ORDER)
            })
            
        for obs in obstacle_features:
            state_values.extend(obs["metrics"])
            state_values.extend(obs["one_hot"])
            
        return np.array(state_values, dtype=np.float32)