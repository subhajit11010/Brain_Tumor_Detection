import shutil
zip_file_path = 'archive.zip'
extraction_path = 'datasets'

shutil.unpack_archive(zip_file_path, extraction_path, 'zip')
print(f"Dataset extracted to {extraction_path}")