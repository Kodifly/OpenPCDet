import os
import argparse

def check_empty_txt_files(directory):
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    print(len(txt_files))
    for txt_file in txt_files:
        path = os.path.join(directory, txt_file)
        if os.stat(path).st_size == 0:
            print(f'File {txt_file} is empty.')
            return False

    print('All .txt files are empty.')
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check if all .txt files in a directory are empty.')
    parser.add_argument('--dir', required=True, help='Directory to check')
    args = parser.parse_args()

    check_empty_txt_files(args.dir)