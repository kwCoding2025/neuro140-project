#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVG to Image and Label Converter for FloorplanCAD Dataset

This script converts SVG files from the FloorplanCAD dataset into:
1. PNG images
2. JSON label files
3. Pickle files containing structured data

Requirements:
- cairosvg (for SVG to PNG conversion)
- svgpathtools (for SVG path parsing)
- numpy
- pickle
- json
"""

import os
import json
import pickle
import logging
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from cairosvg import svg2png
from svgpathtools import parse_path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants
SVG_NS = {"svg": "http://www.w3.org/2000/svg", 
          "inkscape": "http://www.inkscape.org/namespaces/inkscape"}
OUTPUT_IMAGE_SIZE = (512, 512)  # Default output image size

class SVGConverter:
    def __init__(self, input_dir, output_dir, image_size=OUTPUT_IMAGE_SIZE):
        """
        Initialize the SVG converter
        
        Args:
            input_dir: Directory containing SVG files
            output_dir: Directory to save output files
            image_size: Tuple of (width, height) for output images
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        
        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.json_dir = self.output_dir / "json"
        self.pkl_dir = self.output_dir / "pkl"
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)
        os.makedirs(self.pkl_dir, exist_ok=True)
    
    def find_svg_files(self):
        """Find all SVG files in the input directory and subdirectories"""
        svg_files = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.lower().endswith('.svg'):
                    svg_files.append(os.path.join(root, file))
        return svg_files
    
    def convert_svg_to_png(self, svg_file, output_file):
        """Convert SVG file to PNG"""
        try:
            # Convert Path objects to strings for cairosvg
            svg_file_str = str(svg_file)
            output_file_str = str(output_file)
            
            svg2png(url=svg_file_str, 
                   write_to=output_file_str, 
                   output_width=self.image_size[0], 
                   output_height=self.image_size[1],
                   background_color="white")
            return True
        except Exception as e:
            logger.error(f"Error converting {svg_file} to PNG: {str(e)}")
            return False
    
    def extract_paths_from_svg(self, svg_file):
        """Extract path data from SVG file"""
        try:
            tree = ET.parse(svg_file)
            root = tree.getroot()
            
            # Get SVG dimensions
            width = float(root.get('width', '100.0').replace('px', ''))
            height = float(root.get('height', '100.0').replace('px', ''))
            
            # Extract layers and paths
            layers = {}
            for g in root.findall(".//svg:g", SVG_NS):
                layer_id = g.get('id', '')
                layer_label = g.get(f"{{{SVG_NS['inkscape']}}}label", layer_id)
                
                paths = []
                for path in g.findall(".//svg:path", SVG_NS):
                    path_data = {
                        'd': path.get('d', ''),
                        'stroke': path.get('stroke', 'none'),
                        'stroke-width': path.get('stroke-width', '0'),
                        'fill': path.get('fill', 'none'),
                        'semantic-id': path.get('semantic-id', ''),
                        'instance-id': path.get('instance-id', '')
                    }
                    
                    # Parse path to get coordinates
                    try:
                        svg_path = parse_path(path_data['d'])
                        # Extract points along the path
                        points = []
                        for i in range(100):  # Sample 100 points along the path
                            t = i / 99.0
                            point = svg_path.point(t)
                            points.append((point.real / width, point.imag / height))  # Normalize to [0,1]
                        path_data['points'] = points
                    except Exception:
                        # If path parsing fails, just store the raw data
                        path_data['points'] = []
                    
                    paths.append(path_data)
                
                # Only add layers with paths
                if paths:
                    layers[layer_label] = paths
            
            # Extract text elements
            texts = []
            for text in root.findall(".//svg:text", SVG_NS):
                text_data = {
                    'x': float(text.get('x', '0')),
                    'y': float(text.get('y', '0')),
                    'fill': text.get('fill', 'black'),
                    'font-family': text.get('font-family', ''),
                    'font-size': text.get('font-size', ''),
                    'transform': text.get('transform', ''),
                    'content': text.text.strip() if text.text else ''
                }
                texts.append(text_data)
            
            return {
                'width': width,
                'height': height,
                'layers': layers,
                'texts': texts
            }
            
        except Exception as e:
            logger.error(f"Error extracting paths from {svg_file}: {str(e)}")
            return None
    
    def process_svg_file(self, svg_file):
        """Process a single SVG file"""
        try:
            # Get relative path to maintain directory structure
            rel_path = os.path.relpath(str(svg_file), str(self.input_dir))
            base_name = os.path.splitext(rel_path)[0]
            
            # Create output paths
            png_file = self.images_dir / f"{base_name}.png"
            json_file = self.json_dir / f"{base_name}.json"
            pkl_file = self.pkl_dir / f"{base_name}.pkl"
            
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(str(png_file)), exist_ok=True)
            os.makedirs(os.path.dirname(str(json_file)), exist_ok=True)
            os.makedirs(os.path.dirname(str(pkl_file)), exist_ok=True)
            
            # Convert SVG to PNG
            png_success = self.convert_svg_to_png(svg_file, png_file)
            
            # Extract path data
            svg_data = self.extract_paths_from_svg(svg_file)
            
            if svg_data:
                # Save as JSON
                with open(str(json_file), 'w', encoding='utf-8') as f:
                    json.dump(svg_data, f, ensure_ascii=False, indent=2)
                
                # Save as pickle
                with open(str(pkl_file), 'wb') as f:
                    pickle.dump(svg_data, f)
                
                return True, rel_path
            else:
                logger.warning(f"Failed to extract data from {svg_file}")
                return False, rel_path
                
        except Exception as e:
            logger.error(f"Error processing {svg_file}: {str(e)}")
            return False, svg_file
    
    def convert_all(self, num_workers=4):
        """Convert all SVG files to PNG and extract data"""
        svg_files = self.find_svg_files()
        total_files = len(svg_files)
        logger.info(f"Found {total_files} SVG files to process")
        
        success_count = 0
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.process_svg_file, svg_file): svg_file for svg_file in svg_files}
            
            for i, future in enumerate(as_completed(futures), 1):
                success, file_path = future.result()
                if success:
                    success_count += 1
                
                if i % 100 == 0 or i == total_files:
                    logger.info(f"Processed {i}/{total_files} files ({success_count} successful)")
        
        logger.info(f"Conversion complete. Successfully processed {success_count}/{total_files} files.")
        logger.info(f"Images saved to: {self.images_dir}")
        logger.info(f"JSON files saved to: {self.json_dir}")
        logger.info(f"Pickle files saved to: {self.pkl_dir}")

def main():
    parser = argparse.ArgumentParser(description='Convert SVG files to PNG images and extract data')
    parser.add_argument('--input', type=str, default='./floorplancad-dataset',
                        help='Input directory containing SVG files')
    parser.add_argument('--output', type=str, default='./floorplancad-processed',
                        help='Output directory for PNG images and data files')
    parser.add_argument('--width', type=int, default=512,
                        help='Width of output images')
    parser.add_argument('--height', type=int, default=512,
                        help='Height of output images')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of worker processes')
    
    args = parser.parse_args()
    
    converter = SVGConverter(
        input_dir=args.input,
        output_dir=args.output,
        image_size=(args.width, args.height)
    )
    
    converter.convert_all(num_workers=args.workers)

if __name__ == "__main__":
    main() 