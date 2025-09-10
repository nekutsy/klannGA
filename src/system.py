from geometry import Point, Rod
from constraints import Hinge, RigidConnection

class RodSystem:
    __slots__ = ('points', 'rods', 'hinges', 'rigid_connections')
    
    def __init__(self):
        self.points = []
        self.rods = []
        self.hinges = []
        self.rigid_connections = []
    
    def add_point(self, x, y, fixed=False):
        point = Point(x, y, fixed)
        self.points.append(point)
        return point
    
    def add_rod(self, point1, point2, length=None):
        rod = Rod(point1, point2, length)
        self.rods.append(rod)
        return rod
    
    def add_hinge(self, point, rods):
        hinge = Hinge(point, rods)
        self.hinges.append(hinge)
        return hinge
    
    def add_rigid_connection(self, rod1, rod2, angle):
        connection = RigidConnection(rod1, rod2, angle)
        self.rigid_connections.append(connection)
        return connection
    
    def update(self, dt, iterations=5):
        for _ in range(iterations):
            for rod in self.rods:
                rod.update(dt)
            
            for hinge in self.hinges:
                hinge.enforce_constraint()
            
            for rod in self.rods:
                rod.maintain_length()
            
            for connection in self.rigid_connections:
                connection.enforce_constraint()
            
            for rod in self.rods:
                rod.maintain_length()
        
        for point in self.points:
            point.update_position(dt)

class RodSystemBuilder:
    __slots__ = ('system', 'point_dict', 'rod_dict')
    
    def __init__(self):
        self.system = RodSystem()
        self.point_dict = {}
        self.rod_dict = {}
    
    def add_point(self, name, x, y, fixed=False):
        point = self.system.add_point(x, y, fixed)
        self.point_dict[name] = point
        return point
    
    def add_rod(self, name, point1_name, point2_name, length=None):
        point1 = self.point_dict[point1_name]
        point2 = self.point_dict[point2_name]
        rod = self.system.add_rod(point1, point2, length)
        self.rod_dict[name] = rod
        return rod
    
    def add_hinge(self, name, point_name, rod_names):
        point = self.point_dict[point_name]
        rods = [self.rod_dict[rod_name] for rod_name in rod_names]
        hinge = self.system.add_hinge(point, rods)
        return hinge
    
    def add_rigid_connection(self, name, rod1_name, rod2_name, angle):
        rod1 = self.rod_dict[rod1_name]
        rod2 = self.rod_dict[rod2_name]
        connection = self.system.add_rigid_connection(rod1, rod2, angle)
        return connection
    
    def set_angular_velocity(self, rod_name, angular_velocity, pivot_point_name=None):
        rod = self.rod_dict[rod_name]
        pivot_point = self.point_dict[pivot_point_name] if pivot_point_name else None
        rod.set_angular_velocity(angular_velocity, pivot_point)
    
    def get_system(self):
        return self.system