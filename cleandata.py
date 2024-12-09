import shutil
import os
def remove_ipynb_checkpoints(directory):
    for root, dirs, files in os.walk(directory):
        if '.ipynb_checkpoints' in dirs:
            shutil.rmtree(os.path.join(root, '.ipynb_checkpoints'))
            print(f"Removed .ipynb_checkpoints from {root}")

# Remove .ipynb_checkpoints from train, val, and test directories
remove_ipynb_checkpoints('img_dataset/train')
remove_ipynb_checkpoints('img_dataset/val')
remove_ipynb_checkpoints('img_dataset/test')