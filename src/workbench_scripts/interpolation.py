import numpy as np
from scipy.interpolate import interp1d

'''
Created with ChatGPT
>>> To smooth jittering of fly head's centroid
'''


# Sample time series data
timestamps = [0, 1, 2, 3, 4]  # Timestamps
x_values = [1, 3, 2, 4, 1]  # X coordinates
y_values = [2, 1, 3, 2, 4]  # Y coordinates

# Create interpolation functions for x and y coordinates
interp_func_x = interp1d(timestamps, x_values, kind='cubic')
interp_func_y = interp1d(timestamps, y_values, kind='cubic')

# Generate new timestamps for interpolation
new_timestamps = np.linspace(0, 4, num=100)

# Perform interpolation for x and y coordinates
interpolated_x = interp_func_x(new_timestamps)
interpolated_y = interp_func_y(new_timestamps)

# Print the interpolated x and y values
for timestamp, x, y in zip(new_timestamps, interpolated_x, interpolated_y):
    print(f"{timestamp}: {x}, {y}")