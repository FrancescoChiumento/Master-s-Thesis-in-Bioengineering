import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# data_updated = {
#     'Patient': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # Adding an extra patient for the new data
#     'Convergence_Best_Ref': [0.72, 0.66, 0.84, 0.80, 0.80, 0.83, 0.75, 0.70, 0.74, 0.68, 0.77, None],  # None for the 12th patient
#     'Average_Atlas_Best_Ref': [0.80, 0.78, 0.81, 0.79, 0.82, 0.78, 0.82, 0.76, 0.81, 0.77, 0.80, None],  # None for the 12th patient
#     'Convergence_Worst_Ref': [0.69, 0.53, 0.77, 0.74, 0.78, 0.78, 0.60, 0.73, 0.65, 0.71, 0.78, None],  # None for the 12th patient
#     'Average_Atlas_Worst_Ref': [0.81, 0.72, 0.80, 0.82, 0.81, 0.83, 0.79, 0.76, 0.75, 0.78, 0.80, None],  # None for the 12th patient
#     'Average_Atlas_Manually_Segmented': [0.79, 0.77, 0.82, 0.80, 0.80, 0.82, 0.80, 0.76, 0.79, 0.74, 0.78, 0.83]  # New data for the 12th patient included
# }

# data_updated = {
#     'Patient': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # Adding an extra patient for the new data
#     'Convergence_Best_Ref': [0.57, 0.49, 0.72, 0.66, 0.67, 0.71, 0.60, 0.53, 0.59, 0.52, 0.63, None],  # None for the 12th patient
#     'Average_Atlas_Best_Ref': [0.66, 0.64, 0.68, 0.66, 0.70, 0.64, 0.70, 0.61, 0.67, 0.62, 0.66, None],  # None for the 12th patient
#     'Convergence_Worst_Ref': [0.53, 0.36, 0.63, 0.59, 0.63, 0.64, 0.42, 0.57, 0.48, 0.55, 0.64, None],  # None for the 12th patient
#     'Average_Atlas_Worst_Ref': [0.67, 0.56, 0.67, 0.69, 0.68, 0.71, 0.65, 0.61, 0.60, 0.63, 0.67, None],  # None for the 12th patient
#     'Average_Atlas_Manually_Segmented': [0.65, 0.62, 0.70, 0.66, 0.67, 0.70, 0.66, 0.61, 0.66, 0.58, 0.63, 0.71]  # New data for the 12th patient included
# }

data_updated = {
    'Patient': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # Adding an extra patient for the new data
    'Convergence_Best_Ref': [-0.08, -0.36, -0.06, 0.06, 0.12, 0.07, -0.19,-0.19, -0.23, -0.16, 0.10, None],  # None for the 12th patient
    'Average_Atlas_Best_Ref': [0.07, -0.08, 0.04, 0.18, 0.14, 0.24, -0.18, -0.04, -0.05, -0.01, -0.03, None],  # None for the 12th patient
    'Convergence_Worst_Ref': [-0.34, -0.31, -0.19, -0.13, -0.12, -0.26, -0.15, -0.27, -0.34, -0.28, -0.25, None],  # None for the 12th patient
    'Average_Atlas_Worst_Ref': [-0.07, -0.12, -0.00, 0.03, 0.08, -0.19, -0.10, -0.22, 0.10, -0.16, 0.04, None],  # None for the 12th patient
    'Average_Atlas_Manually_Segmented': [-0.01, -0.09, -0.02, 0.12, 0.15, 0.15, -0.08, 0.03, -0.07, 0.01, -0.04, 0.03]  # New data for the 12th patient included
}



# Creating a new DataFrame
df_jaccard_updated = pd.DataFrame(data_updated)

# Melting the DataFrame to work with seaborn
df_melted_updated = df_jaccard_updated.melt(id_vars='Patient', var_name='Method', value_name='VOLUME SIMILARITY ')

# palette_lighter = {
#     "Convergence_Best_Ref": "lightcoral", 
#     "Average_Atlas_Best_Ref": "lightcyan",
#     "Convergence_Worst_Ref": "lightpink",
#     "Average_Atlas_Worst_Ref": "lightgreen",
#     "Average_Atlas_Manually_Segmented": "lightblue"  # Color for the new group
# }

# palette_lighter = {
#     "Convergence_Best_Ref": "azure", 
#     "Average_Atlas_Best_Ref": "lightgrey",
#     "Convergence_Worst_Ref": "thistle",
#     "Average_Atlas_Worst_Ref": "lightyellow",
#     "Average_Atlas_Manually_Segmented": "lightsteelblue"  # Color for the new group
# }

palette_lighter = {
    "Convergence_Best_Ref": "lavenderblush", 
    "Average_Atlas_Best_Ref": "mintcream",
    "Convergence_Worst_Ref": "seashell",
    "Average_Atlas_Worst_Ref": "linen",
    "Average_Atlas_Manually_Segmented": "beige"  # Color for the new group
}

# Creating a single plot with the style close to the provided figure
plt.figure(figsize=(12, 6))  # Adjusted figure size for additional data

sns.violinplot(
    x='Method', 
    y='VOLUME SIMILARITY ', 
    data=df_melted_updated, 
    palette=palette_lighter, 
    inner='quartile', 
    cut=0,
    hue='Method',
    legend=False,
    density_norm='width'
)

sns.stripplot(
    x='Method', 
    y='VOLUME SIMILARITY ', 
    data=df_melted_updated, 
    color='k', 
    size=4, 
    jitter=True
)

# Adjusting the y-axis to match the example figure more closely
plt.ylim(-0.6, 0.6)

# Removing the legend if it's not needed
plt.legend([],[], frameon=False)

# Setting up the title and labels to match the example figure
plt.title('Comparison of Volume similarity Index Across Different Methods')
plt.xlabel('Method')
plt.ylabel('Volume similarity Index')

# Removing the right and top spines to match the example figure
sns.despine()

# Show plot
plt.show()
