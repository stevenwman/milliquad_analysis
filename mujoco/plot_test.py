import matplotlib
matplotlib.use('Agg')  # Use the non-interactive 'Agg' backend
import numpy as np
import matplotlib.pyplot as plt

plt.plot(np.arange(10), np.arange(10) ** 2)

plt.savefig('my_plot.png') # Save the plot to a file
print("Plot saved to my_plot.png")