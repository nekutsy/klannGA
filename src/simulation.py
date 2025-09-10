import numpy as np
from system import RodSystemBuilder
from analysis import analyze_trajectory

def calculate_trajectory(system, target_point_index, steps=500, dt=0.01):
    if target_point_index < 0 or target_point_index >= len(system.points):
        raise ValueError(f"Point index {target_point_index} out of range")
    
    target_point = system.points[target_point_index]
    trajectory = np.empty((steps, 2))
    for i in range(steps):
        system.update(dt)
        trajectory[i] = target_point.position()
    return trajectory

def create_system(x, y1, y2, rod_lengths, rigid_angles):
    builder = RodSystemBuilder()
    
    builder.add_point('p1', 0, 0, fixed=True)
    builder.add_point('p2', x, y1, fixed=True)
    builder.add_point('p3', x, y2, fixed=True)
    builder.add_point('p4', 0, 15)
    builder.add_point('p5', x, -y1+17.7)
    builder.add_point('p6', -45, 30)
    builder.add_point('p7', -50, 0)
    builder.add_point('p8', -60, -55)
    
    builder.add_rod('rod1', 'p1', 'p4', length=15)
    builder.add_rod('rod2', 'p2', 'p6', length=rod_lengths[0])
    builder.add_rod('rod3', 'p3', 'p5', length=rod_lengths[1])
    builder.add_rod('rod4', 'p4', 'p5', length=rod_lengths[2])
    builder.add_rod('rod5', 'p5', 'p7', length=rod_lengths[3])
    builder.add_rod('rod6', 'p6', 'p7', length=rod_lengths[4])
    builder.add_rod('rod7', 'p7', 'p8', length=rod_lengths[5])
    
    builder.add_hinge('hinge1', 'p4', ['rod1', 'rod4'])
    builder.add_hinge('hinge2', 'p5', ['rod3', 'rod4'])
    builder.add_hinge('hinge3', 'p6', ['rod2', 'rod6'])
    builder.add_hinge('hinge4', 'p7', ['rod6', 'rod7'])
    
    builder.add_rigid_connection('rigid1', 'rod4', 'rod5', rigid_angles[0])
    builder.add_rigid_connection('rigid2', 'rod6', 'rod7', rigid_angles[1])
    
    builder.set_angular_velocity('rod1', -0.75, 'p1')
    
    return builder.get_system()

def simulate_single_system(x, y1, y2, rod_lengths, rigid_angles, target_point_index=7, steps=430, dt=0.01):
    try:
        system = create_system(x=x, y1=y1, y2=y2, rod_lengths=rod_lengths, rigid_angles=rigid_angles)
        
        for _ in range(100):
            system.update(dt)
        
        trajectory = calculate_trajectory(system, target_point_index, steps, dt)
        analysis = analyze_trajectory(trajectory)
        
        return analysis, trajectory
    except Exception as e:
        return {'width': np.nan, 'height': np.nan, 'straight_length': np.nan, 
                'straightness': np.nan, 'slope': np.nan, 'x_offset': np.nan, 
                'y_offset': np.nan}, np.array([])