import os
import shutil
extraction_path = 'datasets'
from sklearn.model_selection import train_test_split
training_dir = os.path.join(extraction_path, 'Training')
testing_dir = os.path.join(extraction_path, 'Testing')
output_dir = 'img_dataset'
classes = ['glioma', 'meningioma', 'pituitary', 'notumor']
for split in ['train', 'val', 'test']:
    for class_name in classes:
        os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)
for class_name in classes:
  for class_name in classes:
    class_dir = os.path.join(training_dir, class_name)
    images = os.listdir(class_dir)

    train, val = train_test_split(images, test_size=0.2, random_state=42)

    for image in train:
      if(not image.endswith('.ipynb_checkpoints')):
        shutil.copy(os.path.join(class_dir, image), os.path.join(output_dir, 'train', class_name))
    for image in val:
      if(not image.endswith('.ipynb_checkpoints')):
        shutil.copy(os.path.join(class_dir, image), os.path.join(output_dir, 'val', class_name))

# Copy testing data as is
for class_name in classes:
    test_dir = os.path.join(testing_dir, class_name)
    images = os.listdir(test_dir)

    for image in images:
        shutil.copy(os.path.join(test_dir, image), os.path.join(output_dir, 'test', class_name))
