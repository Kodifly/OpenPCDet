import os
import random

def split_data(directory, train_ratio, val_ratio, test_ratio):
    
    assert int(round(sum([train_ratio, val_ratio, test_ratio]))) == 1, "Ratios must sum to 1"
    
    # Get the list of all .bin files in the 'velodyne' subdirectory
    velodyne_dir = os.path.join(directory, 'velodyne')
    all_files = [f for f in os.listdir(velodyne_dir) if f.endswith('.bin')]
    
    # Randomly shuffle the list
    random.shuffle(all_files)

    # Calculate the number of training, validation, and testing files
    num_train = int(len(all_files) * train_ratio)
    num_val = int(len(all_files) * val_ratio)

    # Split the list into training, validation, and testing sets
    train_files = all_files[:num_train]
    val_files = all_files[num_train:num_train + num_val]
    test_files = all_files[num_train + num_val:]
    
    # Create a new directory called 'ImageSets'
    imagesets_dir = os.path.join(directory, 'ImageSets')
    os.makedirs(imagesets_dir, exist_ok=True)
    
    # Write the training, validation, and testing file lists to disk
    file_lists = {'full.txt': all_files, 'train.txt': train_files, 'val.txt': val_files, 'test.txt': test_files, 'train_val.txt': train_files + val_files}
    for filename, file_list in file_lists.items():
        with open(os.path.join(imagesets_dir, filename), 'w') as f:
            for file in file_list:
                f.write(f"{file.split('.')[0]}\n")

if __name__ == "__main__":
    split_data('/home/gabriel/kodifly-pcdet/data/tko/training', train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)