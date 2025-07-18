import numpy as np 
from PIL import Image 
import cv2 
import os 
import random 

from scipy.ndimage import gaussian_filter 

MULTIPLE_ITER = 5
SIGMA=5

save_path = f'../datasets/MCBM'
os.makedirs(save_path, exist_ok=True)



for img_num in range(1000): 
    MULTIPLE_ITER = random.choice([5,6,7])
    SIGMA = random.choice([15, 25, 35])
    # Parameters for the 2D array
    rows, cols = 256, 256

    # Initialize the 2D array
    Z = np.zeros((rows, cols))

    # Define transition probabilities for Markov Chain
    P = {
        'up': 0.25,
        'down': 0.25,
        'left': 0.25,
        'right': 0.25
    }

    # Initial state for Markov Chain
    # current_row, current_col = rows // 2, cols // 2
    current_row, current_col = np.random.randint(0, rows-1), np.random.randint(0, cols-1)
    Z[current_row, current_col] = 1

    # Simulate the Markov chain with Brownian motion
    np.random.seed(img_num)
    num_points = rows * cols * MULTIPLE_ITER
    increments = np.random.normal(loc=0, scale=1.0, size=(num_points, 2))

    for i in range(num_points):
        r = np.random.rand()
        if r < P['up'] and current_row > 0:
            current_row -= 1
        elif r < P['up'] + P['down'] and current_row < rows - 1:
            current_row += 1
        elif r < P['up'] + P['down'] + P['left'] and current_col > 0:
            current_col -= 1
        elif current_col < cols - 1:
            current_col += 1
        Z[current_row, current_col] += 1
        
        # Add Brownian motion
        current_row = (current_row + int(increments[i, 0])) % rows
        current_col = (current_col + int(increments[i, 1])) % cols
        Z[current_row, current_col] += 1

    # Z_normalized = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))

    # Z_resized = cv2.resize(Z, (1600, 1200))
    
    Z_smoothed = gaussian_filter(Z, sigma=SIGMA)
    # Z_smoothed2 = gaussian_filter(Z_resized, sigma=SIGMA*2)
    # Z_smoothed3 = gaussian_filter(Z_resized, sigma=SIGMA*5)
    
    
    Z_int = SIGMA * Z_smoothed + Z
    Z_normalized = (Z_int - np.min(Z_int)) / (np.max(Z_int) - np.min(Z_int))
    
     
    # Z_normalized = cv2.resize(Z_normalized, (1600, 1200))
    Z_normalized = Image.fromarray((Z_normalized*255).astype(np.uint8))
    Z_normalized.save(os.path.join(save_path, f'{img_num}.png'))