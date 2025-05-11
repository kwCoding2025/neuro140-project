import torch
from torch.utils.data import Dataset
from PIL import Image
import pickle
import logging
import numpy as np
import io
from torchvision import transforms
from pathlib import Path
import cairosvg

logger = logging.getLogger(__name__)

class FloorplanDataset(Dataset):
    def __init__(self, pkl_files, transform=None):
        """
        Args:
            pkl_files: List of paths to pickle files
            transform: Optional transform to be applied on images
        """
        self.pkl_files = pkl_files
        self.transform = transform
        
        # normalize for pretrained
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.pkl_files)

    def __getitem__(self, idx):
        with open(self.pkl_files[idx], 'rb') as f:
            data = pickle.load(f)
            
        # svg to image
        svg_string = self._create_svg_string(data)
        image_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # apply transforms
        if self.transform:
            image = self.transform(image)
        
        # extract features
        features = self._extract_features(data)
        
        return {
            'image': self.normalize(image),
            'features': features
        }

    def _create_svg_string(self, data):
        """Create SVG string from data"""
        svg_template = f'''<?xml version="1.0" encoding="UTF-8"?>
        <svg width="{data['width']}" height="{data['height']}" xmlns="http://www.w3.org/2000/svg">'''
        
        # add paths from layers
        for layer_name, paths in data['layers'].items():
            for path in paths:
                svg_template += f'<path d="{path["d"]}" '
                if 'stroke' in path:
                    svg_template += f'stroke="{path["stroke"]}" '
                if 'stroke-width' in path:
                    svg_template += f'stroke-width="{path["stroke-width"]}" '
                if 'fill' in path:
                    svg_template += f'fill="{path["fill"]}" '
                svg_template += '/>'
        
        svg_template += '</svg>'
        return svg_template

    def _extract_features(self, data):
        """Extract relevant features from data"""
        # count walls
        wall_count = 0
        wall_coords = []
        wall_endpoints = []
        intersections = set()
        room_count = 0
        
        for layer_name, paths in data['layers'].items():
            for path in paths:
                if path.get('semantic-id') == '17':
                    wall_count += 1
                    points = path['points']
                    if points:
                        # extend with points
                        wall_coords.extend(points)
                        # store endpoints
                        wall_endpoints.append((points[0], points[-1]))
        
        # estimate rooms
        intersections = set()
        for i, (start1, end1) in enumerate(wall_endpoints):
            for start2, end2 in wall_endpoints[i+1:]:
                # line intersection check
                px, py = self._line_intersection(start1, end1, start2, end2)
                if px is not None and py is not None:
                    intersections.add((round(px, 3), round(py, 3)))
        
        room_count = max(1, len(intersections) // 3) if intersections else 1
        
        # wall_coords to flat array
        if wall_coords:
            # flatten to 1d
            wall_coords_flat = np.array(wall_coords, dtype=np.float32).flatten()
            
            # pad/trunc to 200
            current_len = len(wall_coords_flat)
            if current_len > 200:
                wall_coords_padded = wall_coords_flat[:200]
            elif current_len < 200:
                # pad with zeros
                wall_coords_padded = np.pad(wall_coords_flat, (0, 200 - current_len), 'constant', constant_values=0)
            else:
                wall_coords_padded = wall_coords_flat
        else:
            # if no walls, zeros
            wall_coords_padded = np.zeros(200, dtype=np.float32)
            
        # ensure shape (200,)
        assert wall_coords_padded.shape == (200,), f"Expected shape (200,), but got {wall_coords_padded.shape}"

        return {
            'room_count': torch.tensor(room_count, dtype=torch.float32),
            'wall_count': torch.tensor(wall_count, dtype=torch.float32),
            # return padded tensor
            'wall_coords': torch.from_numpy(wall_coords_padded)
        }

    def _line_intersection(self, start1, end1, start2, end2):
        """Calculate the intersection point of two line segments"""
        x1, y1 = start1
        x2, y2 = end1
        x3, y3 = start2
        x4, y4 = end2

        denominator = (x4 - x3) * (y2 - y1) - (y4 - y3) * (x2 - x1)
        if denominator == 0:
            return None, None

        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator

        if 0 <= ua <= 1 and 0 <= ub <= 1:
            x = x1 + ua * (x2 - x1)
            y = y1 + ua * (y2 - y1)
            return x, y
        else:
            return None, None 