#!/usr/bin/env python
"""
Test video mapping to patient groups
"""

import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.data import get_patient_groups


def discover_all_video_folders(data_dir: str):
    """Discover all video folders in the data directory"""
    video_folders = []
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item.startswith('data_'):
            video_folders.append(item)
    
    video_folders.sort()
    return video_folders


def test_video_mapping():
    """Test the video to patient group mapping"""
    data_dir = "data/processed"
    
    print("=== Testing Video to Patient Group Mapping ===")
    print()
    
    # Get patient groups
    patient_groups = get_patient_groups()
    print("Patient groups definition:")
    for group, videos in patient_groups.items():
        print(f"  {group}: {videos}")
    print()
    
    # Discover video folders
    video_folders = discover_all_video_folders(data_dir)
    print(f"Found {len(video_folders)} video folders:")
    for folder in video_folders:
        print(f"  - {folder}")
    print()
    
    # Test mapping
    print("Video to Patient Group Mapping:")
    patient_to_videos = {group: [] for group in patient_groups.keys()}
    
    for video_folder in video_folders:
        # Use the complete folder name for matching
        video_id = video_folder
        
        # Find which patient group this video belongs to
        found_group = None
        for patient_group, patient_list in patient_groups.items():
            if video_id in patient_list:
                found_group = patient_group
                break
        
        if found_group:
            patient_to_videos[found_group].append(video_folder)
            print(f"  ✓ {video_folder} -> {found_group}")
        else:
            print(f"  ✗ {video_folder} -> NOT FOUND in any group")
    
    print()
    print("Final mapping summary:")
    for patient_group, videos in patient_to_videos.items():
        print(f"  {patient_group}: {len(videos)} videos - {videos}")
    
    return patient_to_videos


if __name__ == "__main__":
    test_video_mapping() 