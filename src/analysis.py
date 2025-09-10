import numpy as np
from scipy import stats
from scipy.spatial import distance
import math
from scipy import spatial

def is_closed(trajectory, threshold=10.0):
    if len(trajectory) < 2:
        return False
    
    start_point = trajectory[0]
    end_point = trajectory[-1]
    dist = np.linalg.norm(start_point - end_point)
    
    return dist <= threshold

def is_active(trajectory, threshold=1.0):
    if len(trajectory) < 10:
        return False
    last_points = trajectory[-10:]
    movement = np.sum(np.sqrt(np.sum(np.diff(last_points, axis=0)**2, axis=1)))
    return movement > threshold

def find_straight_segment(points, left_margin_percent=0.05, right_margin_percent=0.1):
    if len(points) < 2:
        return None, None, None
    
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    width = max_x - min_x
    height = max_y - min_y
    
    left_bound = min_x + left_margin_percent * width
    right_bound = max_x - right_margin_percent * width
    
    min_y_idx = np.argmin(y_coords)
    min_y_point = points[min_y_idx]
    
    left_idx = min_y_idx
    while left_idx > 0 and points[left_idx - 1, 0] >= left_bound:
        left_idx -= 1
    
    right_idx = min_y_idx
    while right_idx < len(points) - 1 and points[right_idx + 1, 0] <= right_bound:
        right_idx += 1
    
    segment = points[left_idx:right_idx + 1]
    if len(segment) < 2:
        return None, None, None
    
    segment_x = segment[:, 0]
    segment_y = segment[:, 1]
    
    is_monotonic = np.all(np.diff(segment_x) >= 0) or np.all(np.diff(segment_x) <= 0)
    
    if not is_monotonic:
        diff_sign = np.sign(np.diff(segment_x))
        inflection_points = np.where(np.diff(diff_sign) != 0)[0] + 1
        
        inflection_points = np.concatenate(([0], inflection_points, [len(segment)-1]))
        
        max_length = 0
        best_start = 0
        best_end = 0
        
        for i in range(len(inflection_points) - 1):
            start = inflection_points[i]
            end = inflection_points[i+1]
            subsegment = segment[start:end+1]
            
            if len(subsegment) > max_length:
                max_length = len(subsegment)
                best_start = start
                best_end = end
        
        segment = segment[best_start:best_end+1]
        
        if len(segment) < 2:
            return None, None, None
    
    start_point = segment[0]
    end_point = segment[-1]
    
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    segment_length = np.sqrt(dx**2 + dy**2)
    
    if segment_length == 0:
        return None, None, None
    
    A = -dy
    B = dx
    C = start_point[0]*dy - start_point[1]*dx
    
    norm = np.sqrt(A**2 + B**2)
    if norm == 0:
        return None, None, None
        
    distances = np.abs(A*segment[:,0] + B*segment[:,1] + C) / norm
    
    t = ((segment[:,0] - start_point[0])*dx + 
         (segment[:,1] - start_point[1])*dy) / segment_length**2
    
    sorted_indices = np.argsort(t)
    t_sorted = t[sorted_indices]
    dist_sorted = distances[sorted_indices]
    
    area = np.trapz(dist_sorted, t_sorted) * segment_length
    
    max_possible_area = segment_length * np.max(distances) if np.max(distances) > 0 else 0
    normalized_area = area / max_possible_area if max_possible_area > 0 else 0
    
    straightness = max(0, 1 - normalized_area)
    length = segment_length
    
    return length, straightness, segment

def analyze_trajectory(trajectory):
    if len(trajectory) < 2:
        return {
            'width': np.nan,
            'height': np.nan,
            'straight_length': np.nan,
            'straightness': np.nan,
            'slope': np.nan,
            'x_offset': np.nan,
            'y_offset': np.nan,
            'is_closed': False,
        }
    
    try:
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        
        width = np.max(x) - np.min(x)
        height = np.max(y) - np.min(y)
        
        straight_length, straightness, straight_segment = find_straight_segment(trajectory)
        
        if straight_segment is not None:
            slope, r_squared = calculate_slope(straight_segment)
            x_offset = np.mean(straight_segment[:, 0])
            y_offset = np.mean(straight_segment[:, 1])
        else:
            straight_length = straightness = slope = x_offset = y_offset = np.nan

        result = {
            'width': width,
            'height': height,
            'straight_length': straight_length,
            'straightness': straightness,
            'slope': slope,
            'x_offset': x_offset,
            'y_offset': y_offset,
            'is_closed': is_closed(trajectory)
        }
        
        return result
    except Exception as e:
        return {
            'width': np.nan,
            'height': np.nan,
            'straight_length': np.nan,
            'straightness': np.nan,
            'slope': np.nan,
            'x_offset': np.nan,
            'y_offset': np.nan,
            'is_closed': False
        }

def calculate_slope(segment):
    if len(segment) < 2:
        return np.nan, np.nan
        
    try:
        start_point = segment[0]
        end_point = segment[-1]
        
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg, 1.0
    except:
        return np.nan, np.nan