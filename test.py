import numpy as np
import matplotlib.pyplot as plt

# Define your data
a = [[1, 2, 3, 4, 1, 2],
     [234, 1, 324, 32, 11, 3],
     [12, 4, 22, 7, 1, 34]]

# Calculate the mean and standard error for each sublist using NumPy
means = [np.mean(sublist) for sublist in a]
std_errors = [np.std(sublist, ddof=1) / np.sqrt(len(sublist)) for sublist in a]

# Create a line plot to connect the mean values
x_labels = ['Sublist 1', 'Sublist 2', 'Sublist 3']
x_pos = np.arange(len(x_labels))

plt.errorbar(x_pos, means, yerr=std_errors, fmt='o-', capsize=5, label='Mean with SE')
plt.xticks(x_pos, x_labels)
plt.xlabel('Sublist')
plt.ylabel('Value')
plt.title('Mean Value with Standard Error')
plt.legend()
plt.grid()
plt.show()
