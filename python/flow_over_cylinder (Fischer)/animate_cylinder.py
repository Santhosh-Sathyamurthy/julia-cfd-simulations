# MIT License
# Copyright (c) 2025 Santhosh S
# See LICENSE file for full license text.

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

# Configuration
ver = 1
Re_value = 600
plot_types = ["vorticity", "velocity"]
dpi = 200
cylinder_type = "dual"
simulation_duration = 30  # seconds of final video duration

def make_video(plot_type):
    output_dir = f"v{str(ver)}_{cylinder_type}_re_{str(Re_value)}/{plot_type}_frames/"
    output_file = f"outputs/mp4/v{str(ver)}_{cylinder_type}_{plot_type}_re_{str(Re_value)}.mp4"

    # Set up figure
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='black')
    ax.set_axis_off()  # Hide axes for cleaner visualization

    # Get list of frame files
    frame_path = Path(output_dir)
    frame_files = sorted(frame_path.glob(f"{plot_type}_frame_*.png"))  # Sort by filename
    if not frame_files:
        raise FileNotFoundError(f"No frames found for '{plot_type}' in {output_dir}")

    num_frames = len(frame_files)
    fps = num_frames / simulation_duration  # <-- Automated FPS

    print(f"Found {num_frames} frames for '{plot_type}' in {output_dir}")
    print(f"Computed FPS for {simulation_duration}s video: {fps:.3f}")

    # Load first frame to initialize
    img = Image.open(frame_files[0])
    img_array = np.array(img)
    im = ax.imshow(img_array, animated=True)

    # Initialize tqdm progress bar
    progress_bar = tqdm(total=num_frames, desc=f"Processing {plot_type} frames", unit="frame")

    def update(frame_file):
        """Update function for animation."""
        img = Image.open(frame_file)
        img_array = np.array(img)
        im.set_array(img_array)
        progress_bar.update(1)  # Update progress bar
        return [im]

    # Create animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frame_files,
        interval=1000 / fps,
        blit=True
    )

    # Save animation
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    ani.save(output_file, writer=writer, dpi=dpi)
    print(f"Animation saved as {output_file}")

    progress_bar.close()  # Close progress bar
    plt.close(fig)

if __name__ == "__main__":
    for plot_type in plot_types:
        make_video(plot_type)