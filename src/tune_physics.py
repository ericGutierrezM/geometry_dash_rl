import matplotlib.pyplot as plt

def plot_jump(gravity, jump_force, game_speed, ground_y=480):
    y = ground_y
    vy = jump_force
    x = 0
    
    trajectory_x = [x]
    trajectory_y = [y]
    
    frames_in_air = 0
    
    while True:
        frames_in_air += 1
        vy += gravity
        y += vy
        x += game_speed
        
        if y >= ground_y:
            y = ground_y
            trajectory_x.append(x)
            trajectory_y.append(y)
            break
            
        trajectory_x.append(x)
        trajectory_y.append(y)
        
    print(f"Total frames in air: {frames_in_air}")
    print(f"Max height reached: {ground_y - min(trajectory_y):.1f} pixels")
    print(f"Total horizontal distance covered: {x:.1f} pixels")
        
    plt.figure(figsize=(10, 4))
    plt.plot(trajectory_x, trajectory_y, marker='o', color='red', linestyle='--')
    
    # Invert Y axis because in screen coordinates, Y=0 is the top
    plt.gca().invert_yaxis() 
    plt.axhline(y=ground_y, color='black', linestyle='-', label="Ground")
    
    plt.title(f"Jump Arc: g={gravity}, j={jump_force}, speed={game_speed}")
    plt.xlabel("Horizontal Distance (Pixels)")
    plt.ylabel("Vertical Position (Pixels)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Tweak these three numbers until the printed output matches your video frame counting!
plot_jump(gravity=1.09375, jump_force=-18.046875, game_speed=7.8125)