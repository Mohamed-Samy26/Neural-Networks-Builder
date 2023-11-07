
import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the Excel sheet
data = pd.read_excel('Dry_Bean_Dataset.xlsx')

# Get the first and second features
feature1 = data['MinorAxisLength']
feature2 = data['roundnes']

# Get the class labels
class_labels = data['Class']

# Create a scatter plot
colors = {'BOMBAY': 'blue', 'CALI': 'green', 'SIRA': 'red'}
for class_label in class_labels.unique():
    plt.scatter(feature1[class_labels == class_label],
                feature2[class_labels == class_label],
                c=colors[class_label],
                label=class_label)

# Add labels to the axes
plt.xlabel('MinorAxisLength')
plt.ylabel('roundnes')

# Add a title to the plot
plt.title('Scatter Plot of MinorAxisLength vs roundnes')

# Add a legend
plt.legend()

# Show the plot
plt.show()