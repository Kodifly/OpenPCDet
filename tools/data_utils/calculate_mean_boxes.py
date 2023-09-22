import os
import pandas as pd

# Path to the directory
base_dir = "/home/gabriel/kodifly-pcdet/data/vod_lidar"
training_dir = os.path.join(base_dir, "training", "label_2")
image_sets_dir = os.path.join(base_dir, "ImageSets")

# Load file names from train.txt
with open(os.path.join(image_sets_dir, "train.txt"), 'r') as f:
    train_files = [line.strip() for line in f]

# Initialize a list to store the data
data = []

# Loop through all .txt files in the directory
for file_name in train_files:
    with open(os.path.join(training_dir, file_name + ".txt"), 'r') as f:
        for line in f:
            # Split the line into parts
            parts = line.split()
            # Parse the class and dimensions
            class_name = parts[0]
            h, w, l = map(float, parts[8:11])
            # Append the data to the list
            data.append({"class": class_name, "h": h, "w": w, "l": l})

# Convert the list into a DataFrame
df = pd.DataFrame(data)

# Calculate averages for each class
averages = df.groupby("class").mean().reset_index()

# Get the unique class names
classes = df['class'].unique()

# Initialize a list to store the mean sizes
mean_sizes = []

# Loop through the classes and append the mean sizes
for class_name in classes:
    row = averages[averages['class'] == class_name]
    if not row.empty:
        mean_sizes.append([class_name, row.iloc[0]['l'], row.iloc[0]['w'], row.iloc[0]['h']])

# Print the list of mean sizes
for mean_size in mean_sizes:
    print(mean_size)