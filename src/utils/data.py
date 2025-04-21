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
        
        # Standard normalization for pretrained models
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.pkl_files)

    def __getitem__(self, idx):
        with open(self.pkl_files[idx], 'rb') as f:
            data = pickle.load(f)
            
        # Convert SVG to image using cairosvg
        svg_string = self._create_svg_string(data)
        image_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Extract features directly from the data
        features = self._extract_features(data)
        
        return {
            'image': self.normalize(image),
            'features': features
        }

    def _create_svg_string(self, data):
        """Create SVG string from data"""
        svg_template = f'''<?xml version="1.0" encoding="UTF-8"?>
        <svg width="{data['width']}" height="{data['height']}" xmlns="http://www.w3.org/2000/svg">'''
        
        # Add all paths from layers
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
        # Count walls (semantic-id "17")
        wall_count = 0
        wall_coords = []
        wall_endpoints = []
        intersections = set() # Initialize intersections as an empty set
        room_count = 0
        
        for layer_name, paths in data['layers'].items():
            for path in paths:
                if path.get('semantic-id') == '17':
                    wall_count += 1
                    points = path['points']
                    if points:  # Ensure we have points
                        # Extend with individual [x, y] points
                        wall_coords.extend(points) 
                        # Store endpoints for intersection detection
                        wall_endpoints.append((points[0], points[-1]))
        
        # Estimate room count from wall intersections
        intersections = set() # Re-initialize here before calculation loop
        for i, (start1, end1) in enumerate(wall_endpoints):
            for start2, end2 in wall_endpoints[i+1:]:
                # Simple line intersection check
                px, py = self._line_intersection(start1, end1, start2, end2)
                if px is not None and py is not None:
                    intersections.add((round(px, 3), round(py, 3)))
        
        # Estimate room count from intersection points
        # Typically, a room needs at least 3 intersections
        # Ensure room_count calculation happens before return
        room_count = max(1, len(intersections) // 3) if intersections else 1 # Now intersections will always exist
        
        # Convert wall_coords list to a flat numpy array and pad/truncate
        if wall_coords:
            # Flatten the list of [x, y] pairs into a 1D array
            wall_coords_flat = np.array(wall_coords, dtype=np.float32).flatten()
            
            # Pad or truncate to exactly 200 values
            current_len = len(wall_coords_flat)
            if current_len > 200:
                wall_coords_padded = wall_coords_flat[:200]
            elif current_len < 200:
                # Pad with zeros if shorter than 200
                wall_coords_padded = np.pad(wall_coords_flat, (0, 200 - current_len), 'constant', constant_values=0)
            else:
                # Already the correct length
                wall_coords_padded = wall_coords_flat
        else:
            # If no walls were found, create an array of 200 zeros
            wall_coords_padded = np.zeros(200, dtype=np.float32)
            
        # Ensure the final array has shape (200,)
        assert wall_coords_padded.shape == (200,), f"Expected shape (200,), but got {wall_coords_padded.shape}"

        return {
            'room_count': torch.tensor(room_count, dtype=torch.float32),
            'wall_count': torch.tensor(wall_count, dtype=torch.float32),
            # Return the padded/truncated flat tensor
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
            return None, None  # Lines are parallel

        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator

        if 0 <= ua <= 1 and 0 <= ub <= 1:
            x = x1 + ua * (x2 - x1)
            y = y1 + ua * (y2 - y1)
            return x, y
        else:
            return None, None  # Lines do not intersect within the segments 