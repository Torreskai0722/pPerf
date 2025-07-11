#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import deque
import time
import os

class ImageTimeAnalyzer(Node):
    def __init__(self):
        super().__init__('image_time_analyzer')
        
        # Create subscriber for CAM_FRONT images
        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_data/CAM_FRONT',
            self.image_callback,
            1
        )
        
        # Store timestamps and time differences
        self.timestamps = []
        self.time_diffs = deque(maxlen=1000)  # Store last 1000 differences
        self.last_timestamp = None
        
        # Create timer for periodic statistics update
        self.timer = self.create_timer(1.0, self.print_statistics)  # Update every second
        
        self.get_logger().info('Image time analyzer node started')
    
    def image_callback(self, msg):
        current_time = time.time()
        
        if self.last_timestamp is not None:
            time_diff = current_time - self.last_timestamp
            self.time_diffs.append(time_diff)
            
        self.last_timestamp = current_time
        self.timestamps.append(current_time)
    
    def print_statistics(self):
        if len(self.time_diffs) > 0:
            time_diffs_array = np.array(self.time_diffs)
            mean_diff = np.mean(time_diffs_array)
            std_diff = np.std(time_diffs_array)
            min_diff = np.min(time_diffs_array)
            max_diff = np.max(time_diffs_array)
            
            self.get_logger().info(
                f'Time difference statistics (seconds):\n'
                f'Mean: {mean_diff:.3f}\n'
                f'Std: {std_diff:.3f}\n'
                f'Min: {min_diff:.3f}\n'
                f'Max: {max_diff:.3f}\n'
                f'Number of samples: {len(self.time_diffs)}'
            )
    
    def create_final_visualization(self):
        """Create and save a final visualization of the complete distribution."""
        if len(self.time_diffs) > 0:
            time_diffs_array = np.array(self.time_diffs)
            mean_diff = np.mean(time_diffs_array)
            std_diff = np.std(time_diffs_array)
            
            # Create a new figure for the final visualization
            plt.figure(figsize=(12, 8))
            
            # Create subplot for histogram
            plt.subplot(2, 1, 1)
            n, bins, patches = plt.hist(time_diffs_array, bins=50, alpha=0.7)
            
            # Calculate percentages
            total_samples = len(time_diffs_array)
            percentages = (n / total_samples) * 100
            
            # Add count and percentage values on top of each bar
            for i in range(len(patches)):
                if n[i] > 0:  # Only add text if there's data in the bin
                    plt.text(bins[i] + (bins[i+1] - bins[i])/2, n[i],
                            f'n={int(n[i])}\n({percentages[i]:.1f}%)',
                            ha='center', va='bottom')
            
            plt.axvline(mean_diff, color='r', linestyle='dashed', linewidth=1, 
                       label=f'Mean: {mean_diff:.3f}s')
            plt.xlabel('Time Difference (seconds)')
            plt.ylabel('Count')
            plt.title('Distribution of Time Differences Between Consecutive CAM_FRONT Frames')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Create subplot for cumulative distribution
            plt.subplot(2, 1, 2)
            plt.hist(time_diffs_array, bins=50, alpha=0.7, cumulative=True, density=True)
            plt.axvline(mean_diff, color='r', linestyle='dashed', linewidth=1, 
                       label=f'Mean: {mean_diff:.3f}s')
            plt.xlabel('Time Difference (seconds)')
            plt.ylabel('Cumulative Probability')
            plt.title('Cumulative Distribution of Time Differences')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Adjust layout and save
            plt.tight_layout()
            
            # Create output directory if it doesn't exist
            os.makedirs('time_analysis_results', exist_ok=True)
            
            # Save the figure
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            plt.savefig(f'time_analysis_results/time_diff_distribution_{timestamp}.png')
            plt.close()  # Close the figure to free memory
            
            # Print statistics to a text file
            with open(f'time_analysis_results/time_diff_stats_{timestamp}.txt', 'w') as f:
                f.write(f'Time Difference Statistics (seconds):\n')
                f.write(f'Mean: {mean_diff:.3f}\n')
                f.write(f'Std: {std_diff:.3f}\n')
                f.write(f'Min: {np.min(time_diffs_array):.3f}\n')
                f.write(f'Max: {np.max(time_diffs_array):.3f}\n')
                f.write(f'Number of samples: {len(self.time_diffs)}\n')
                f.write(f'95th percentile: {np.percentile(time_diffs_array, 95):.3f}\n')
                f.write(f'99th percentile: {np.percentile(time_diffs_array, 99):.3f}\n')
            
            self.get_logger().info(f'Final visualization saved to time_analysis_results/time_diff_distribution_{timestamp}.png')
            self.get_logger().info(f'Statistics saved to time_analysis_results/time_diff_stats_{timestamp}.txt')

def main(args=None):
    rclpy.init(args=args)
    node = ImageTimeAnalyzer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Creating final visualization...')
        node.create_final_visualization()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

