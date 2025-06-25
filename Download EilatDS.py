import re

def extract_filenames_from_file(filepath):
    """
    Opens a text file, reads its content, and extracts a list of filenames
    based on the pattern '&files[]=filename.flac'.

    Args:
        filepath (str): The path to the text file.

    Returns:
        list: A list of filenames extracted from the file, or None if the file
              cannot be opened or no filenames are found.
    """
    try:
        with open(filepath, 'r') as file:
            content = file.read()
            # Use regular expression to find filenames after '&files[]=' and ending with '.flac'
            filenames = re.findall(r'&files\[\]=(.*?\.flac)', content)
            return filenames
    except FileNotFoundError:
        print(f"Error: File not found at path: {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage:
file_path = '../file_list.txt'  # Replace 'your_file.txt' with the actual path to your file

extracted_file_list = extract_filenames_from_file(file_path)


base_url = "https://csms-acoustic.haifa.ac.il/index.php/s/2UmUoK80Izt0Roe/download?path=%2FData&files[]="

batch_size = 6

import os
import requests
saving_folder = "EilatDS"
if not os.path.exists(saving_folder):
    os.makedirs(saving_folder)

# Process files in groups of batch_size 
for i in range(0, len(extracted_file_list), batch_size ):

    # Take up to batch_size  files for this batch
    batch_files = extracted_file_list[i:i+batch_size ]
    
    # Check which files already exist
    all_files_exist = True
    for file in batch_files:
        file_name = file.split('/')[-1]
        file_path = os.path.join(saving_folder, file_name)
        if not os.path.exists(file_path):
            all_files_exist = False
            break
    
    # Skip batch if all files already exist
    if all_files_exist:
        print(f"Skipping batch {i//batch_size  + 1} (all files already downloaded)")
        continue
    
    # Construct URL with multiple files
    file_url = base_url
    for file in batch_files:
        file_url += file + "&files[]="
    file_url = file_url[:-9]  # Remove the trailing "&files[]="
    
    print(f"Downloading batch {i//batch_size  + 1} with {len(batch_files)} files")
    print(f"URL: {file_url}")
    
    # Download the batch
    response = requests.get(file_url)
    
    # Save each file from the batch, skipping those that already exist
    for file in batch_files:
        file_name = file.split('/')[-1]
        file_path = os.path.join(saving_folder, file_name)
        if os.path.exists(file_path):
            print(f"Skipping {file_name} (already downloaded)")
        else:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Saved {file_name}")