import os
label_folder_path = "/data/LOCO_annotated/untransformed/02081646/labels/"

labels = [label_folder_path + x for x in os.listdir(label_folder_path) if x.endswith(".txt")]

for label_file in labels:
    with open(label_file, "r") as f:
        lines = f.readlines()
    
        new_lines = []
        for line in lines:
            print("old", line)
            line = line.strip().split(' ')
            dz, dy, dx, x, y, z, r = line[-7:]
            new_line = ""
            for i in range(len(line) - 7):
                new_line += line[i] + " "
            new_line += dx + " " + dy + " " + dz + " " + x + " " + y + " " + z + " " + r
            new_lines.append(new_line)
            print("new", new_line)
    
    with open(label_file, "w") as file:
        file.write("\n".join(new_lines))