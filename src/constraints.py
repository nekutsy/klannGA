import numpy as np

class Hinge:
    __slots__ = ('point', 'rods')
    
    def __init__(self, point, rods):
        self.point = point
        self.rods = rods
    
    def enforce_constraint(self):
        for rod in self.rods:
            if rod.point1 == self.point:
                rod.point1.x = self.point.x
                rod.point1.y = self.point.y
            elif rod.point2 == self.point:
                rod.point2.x = self.point.x
                rod.point2.y = self.point.y

class RigidConnection:
    __slots__ = ('rod1', 'rod2', 'angle', 'common_point', 'other_point1', 'other_point2')
    
    def __init__(self, rod1, rod2, angle):
        self.rod1 = rod1
        self.rod2 = rod2
        self.angle = angle
        self._find_common_points()
    
    def _find_common_points(self):
        if self.rod1.point1 == self.rod2.point1 or self.rod1.point1 == self.rod2.point2:
            self.common_point = self.rod1.point1
            self.other_point1 = self.rod1.point2
        else:
            self.common_point = self.rod1.point2
            self.other_point1 = self.rod1.point1
        
        if self.rod2.point1 == self.common_point:
            self.other_point2 = self.rod2.point2
        else:
            self.other_point2 = self.rod2.point1
    
    def enforce_constraint(self):
        if self.other_point2.fixed:
            return
            
        vec1_x = self.other_point1.x - self.common_point.x
        vec1_y = self.other_point1.y - self.common_point.y
        vec2_x = self.other_point2.x - self.common_point.x
        vec2_y = self.other_point2.y - self.common_point.y
        
        vec1_len = np.sqrt(vec1_x*vec1_x + vec1_y*vec1_y)
        vec2_len = np.sqrt(vec2_x*vec2_x + vec2_y*vec2_y)
        
        if vec1_len > 1e-6 and vec2_len > 1e-6:
            vec1_x /= vec1_len
            vec1_y /= vec1_len
            vec2_x /= vec2_len
            vec2_y /= vec2_len
            
            current_angle = np.arctan2(vec2_y, vec2_x) - np.arctan2(vec1_y, vec1_x)
            current_angle = (current_angle + np.pi) % (2 * np.pi) - np.pi
            
            angle_diff = self.angle - current_angle
            
            if abs(angle_diff) < 1e-6:
                return
                
            rotation_matrix = np.array([
                [np.cos(angle_diff), -np.sin(angle_diff)],
                [np.sin(angle_diff), np.cos(angle_diff)]
            ])
            
            new_vec2 = rotation_matrix.dot([vec2_x * vec2_len, vec2_y * vec2_len])
            
            self.other_point2.x = self.common_point.x + new_vec2[0]
            self.other_point2.y = self.common_point.y + new_vec2[1]