import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

# Ground truth and all measurement methods data
ground_truth = [19075, 19366, 19390, 14075, 11737, 16434, 21300, 22953, 26892, 24179, 18258, 18680]
convergence_study_best_reference = [17578, 13520, 18234, 14886, 13243, 17604, 17521, 19025, 21422, 20515, 16539, 18680]
average_atlas_best_reference = [20526, 17954, 20180, 16865, 13553, 20967, 17827, 22080, 25665, 23887, 17786, 22185]
convergence_worst_reference = [13498, 14156, 16083, 12310, 11953, 14573, 16395, 19685, 20514, 17142, 13729, 14577]
average_atlas_worst_reference = [17780, 17124, 19362, 14537, 13685, 17881, 17550, 20862, 21587, 21849, 15597, 19402]
average_atlas_manually_segmented = [18799, 17667, 18915, 15939, 13636, 19171, 19676, 23617, 25068, 24454, 17539, 19312]

# Convert all lists to a DataFrame
df = pd.DataFrame({
    'Ground Truth': ground_truth,
    'Convergence Study Best Reference': convergence_study_best_reference,
    'Average Atlas With Best Reference': average_atlas_best_reference,
    'Convergence Worst Reference': convergence_worst_reference,
    'Average Atlas Worst Reference': average_atlas_worst_reference,
    'Average Atlas Manually Segmented': average_atlas_manually_segmented
})

# Calculate Pearson correlation coefficients for all methods
pearson_coefs = {method: pearsonr(df['Ground Truth'], df[method])[0] for method in df.columns if method != 'Ground Truth'}

# Plotting the scatter plot for all methods
plt.figure(figsize=(12, 10))

# Scatter plots for each method
for method, data in df.items():
    if method != 'Ground Truth':
        plt.scatter(df['Ground Truth'], data, alpha=0.5, label=f'{method} (r={pearson_coefs[method]:.3f})')

# Lines of best fit for each method
for method in df.columns[1:]:
    fit = np.polyfit(df['Ground Truth'], df[method], 1)
    plt.plot(df['Ground Truth'], np.polyval(fit, df['Ground Truth']), label=f'Fit for {method}')

# Diagonal line (x=y)
plt.plot([min(df['Ground Truth']), max(df['Ground Truth'])], 
         [min(df['Ground Truth']), max(df['Ground Truth'])], 
         color='grey', linestyle='--')

plt.gca().set_aspect('equal', adjustable='box')

# Labels, title and grid
plt.xlabel('Ground Truth Volume [cm^3]')
plt.ylabel('Predicted Volume [cm^3]')
plt.title('Volume Correlation')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
