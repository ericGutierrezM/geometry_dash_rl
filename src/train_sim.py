import os
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from geometry_dash_sim import GeometryDashSim

def main():
    # 1. Instanciar el entorno matemático
    print("Cargando el simulador matemático...")
    env = GeometryDashSim()

    # --- NUEVO: Configurar el Logger ---
    # Esto fuerza a la IA a guardar su progreso en consola Y en un archivo CSV
    log_dir = "./sim_logs/"
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "csv"])

    # 2. Configurar el cerebro de la IA (PPO)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,  
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,          
        verbose=1,            
        ent_coef=0.01
    )

    # --- NUEVO: Asignar el logger al modelo ---
    model.set_logger(new_logger)

    # 3. Lanzar el entrenamiento hiperbólico
    TIMESTEPS = 500000  
    print(f"¡Arrancando entrenamiento de {TIMESTEPS} pasos a la velocidad de la luz!")
    
    model.learn(total_timesteps=TIMESTEPS)

    # 4. Guardar el cerebro experto
    model_name = "ppo_gd_simulado_experto"
    model.save(model_name)
    print(f"¡ENTRENAMIENTO TERMINADO! Modelo guardado como: {model_name}.zip")

    # --- NUEVO: Generar las Gráficas ---
    print("Generando gráficas de entrenamiento...")
    csv_path = os.path.join(log_dir, "progress.csv")
    
    if os.path.exists(csv_path):
        # Leer los datos de entrenamiento
        df = pd.read_csv(csv_path)
        
        # Crear una figura con 3 subgráficas (una encima de la otra)
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        
        # 1. Recompensa Media (ep_rew_mean)
        if 'rollout/ep_rew_mean' in df.columns:
            # Usamos dropna() porque 'rollout' y 'train' se actualizan en distintos momentos
            data = df[['time/total_timesteps', 'rollout/ep_rew_mean']].dropna()
            axs[0].plot(data['time/total_timesteps'], data['rollout/ep_rew_mean'], color='green', linewidth=2)
            axs[0].set_title('Recompensa Media (Supervivencia)')
            axs[0].set_ylabel('ep_rew_mean')
            axs[0].grid(True)
        
        # 2. Pérdida de Entropía (entropy_loss)
        if 'train/entropy_loss' in df.columns:
            data = df[['time/total_timesteps', 'train/entropy_loss']].dropna()
            axs[1].plot(data['time/total_timesteps'], data['train/entropy_loss'], color='blue', linewidth=2)
            axs[1].set_title('Pérdida de Entropía (Exploración vs Certeza)')
            axs[1].set_ylabel('entropy_loss')
            axs[1].grid(True)
            
        # 3. Varianza Explicada (explained_variance)
        if 'train/explained_variance' in df.columns:
            data = df[['time/total_timesteps', 'train/explained_variance']].dropna()
            axs[2].plot(data['time/total_timesteps'], data['train/explained_variance'], color='purple', linewidth=2)
            axs[2].set_title('Varianza Explicada (Entendimiento de la Matriz)')
            axs[2].set_xlabel('Timesteps')
            axs[2].set_ylabel('explained_variance')
            axs[2].grid(True)
            
        # Ajustar los espacios y guardar la imagen
        plt.tight_layout()
        plt.savefig("training_metrics.png")
        print("¡Éxito! Abre el archivo 'training_metrics.png' para ver tu gráfica.")
    else:
        print("Error: No se encontró el archivo CSV de logs para graficar.")

if __name__ == "__main__":
    main()