import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the time series data
time_series_file = 'harvard_oxford_time_series.txt'
time_series_data = np.loadtxt(time_series_file)

# Calculate the correlation matrix
correlation_matrix = np.corrcoef(time_series_data.T)

# Save the correlation matrix to a file
np.savetxt('harvard_oxford_correlation_matrix.txt', correlation_matrix, fmt='%.6f')

# Set up the figure and the heatmap using seaborn
plt.figure(figsize=(12, 12))
sns.set(style="white")
sns.heatmap(correlation_matrix, cmap="coolwarm", square=True, center=0, vmin=-1, vmax=1, linewidths=.5)

# Customize the plot (optional)
plt.title('Harvard-Oxford Atlas Functional Connectivity')
plt.xlabel('Regions')
plt.ylabel('Regions')

# Save the heatmap as an image (optional)
plt.savefig('harvard_oxford_correlation_matrix_heatmap.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

