import numpy as np
import matplotlib.pyplot as plt

# Define the x range from 1 to 1000
x = np.arange(1, 1001)

# Define a custom function to create the desired curve
def mountain_curve(x):
    return 5 * np.exp(-(x - 500)**2 / (2 * 100**2)) + 25

# Generate the y values using the custom function
y = mountain_curve(x)

# Plot the curve
plt.plot(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Mountain-like Curve (Increase, Constant, Decrease)')
plt.show()

with open('mountain_curve.txt', 'w') as file:
    for value in y:
        file.write(f'{value}\n')