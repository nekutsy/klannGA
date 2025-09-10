import numpy as np

class Point:
    __slots__ = ('x', 'y', 'fixed', 'velocity', 'connected_rods')
    
    def __init__(self, x, y, fixed=False):
        self.x = x
        self.y = y
        self.fixed = fixed
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.connected_rods = []
    
    def position(self):
        return (self.x, self.y)
    
    def update_position(self, dt):
        if not self.fixed:
            self.x += self.velocity[0] * dt
            self.y += self.velocity[1] * dt

class Rod:
    __slots__ = ('point1', 'point2', 'length', 'angular_velocity', 'pivot_point')
    
    def __init__(self, point1, point2, length=None):
        self.point1 = point1
        self.point2 = point2
        self.length = length if length is not None else self.calculate_length()
        self.angular_velocity = 0.0
        self.pivot_point = None
        
        point1.connected_rods.append(self)
        point2.connected_rods.append(self)
    
    def calculate_length(self):
        dx = self.point2.x - self.point1.x
        dy = self.point2.y - self.point1.y
        return np.sqrt(dx*dx + dy*dy)
    
    def set_angular_velocity(self, angular_velocity, pivot_point=None):
        self.angular_velocity = angular_velocity
        self.pivot_point = pivot_point if pivot_point else self.point1
    
    def update(self, dt):
        if self.angular_velocity != 0 and self.pivot_point:
            if self.pivot_point == self.point1:
                fixed_point = self.point1
                moving_point = self.point2
            else:
                fixed_point = self.point2
                moving_point = self.point1
            
            if moving_point.fixed:
                return
                
            dx = moving_point.x - fixed_point.x
            dy = moving_point.y - fixed_point.y
            
            current_angle = np.arctan2(dy, dx)
            new_angle = current_angle + self.angular_velocity * dt
            
            moving_point.x = fixed_point.x + self.length * np.cos(new_angle)
            moving_point.y = fixed_point.y + self.length * np.sin(new_angle)
            
            moving_point.velocity[0] = -self.length * self.angular_velocity * np.sin(new_angle)
            moving_point.velocity[1] = self.length * self.angular_velocity * np.cos(new_angle)
    
    def maintain_length(self):
        if self.point1.fixed and self.point2.fixed:
            return
        
        # Recalculate actual length
        dx = self.point2.x - self.point1.x
        dy = self.point2.y - self.point1.y
        current_length = np.sqrt(dx*dx + dy*dy)
        
        if abs(current_length - self.length) < 1e-6:
            return
            
        # Calculate correction
        correction_factor = (self.length - current_length) / current_length
        
        if self.point1.fixed:
            self.point2.x += dx * correction_factor
            self.point2.y += dy * correction_factor
        elif self.point2.fixed:
            self.point1.x -= dx * correction_factor
            self.point1.y -= dy * correction_factor
        else:
            half_correction = correction_factor / 2
            self.point1.x -= dx * half_correction
            self.point1.y -= dy * half_correction
            self.point2.x += dx * half_correction
            self.point2.y += dy * half_correction