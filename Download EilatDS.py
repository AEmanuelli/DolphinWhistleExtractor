"""Download the Eilat dolphin dataset (FLAC recordings) from the Haifa CSMS server.

The list of files is read from a text file containing the server's download
links (every ``&files[]=<name>.flac`` occurrence is picked up). Files are
requested in small batches; when several files are requested at once the
server answers with a ZIP archive, which is unpacked into the output folder.

Usage:
    python "Download EilatDS.py" [--file_list ../file_list.txt] [--output EilatDS] [--batch_size 6]
"""
import argparse
import io
import os
import re
import zipfile

BASE_URL = "https://csms-acoustic.haifa.ac.il/index.php/s/2UmUoK80Izt0Roe/download?path=%2FData"


def extract_filenames_from_file(filepath):
    """Return the file names found after '&files[]=' (ending with '.flac') in a text file."""
    with open(filepath, 'r') as file:
        return re.findall(r'&files\[\]=(.*?\.flac)', file.read())


def save_response(content, batch_files, saving_folder):
    """Write a server answer to disk: a ZIP of several files, or a single file."""
    if zipfile.is_zipfile(io.BytesIO(content)):
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                target = os.path.join(saving_folder, os.path.basename(member.filename))
                with archive.open(member) as source, open(target, 'wb') as destination:
                    destination.write(source.read())
                print(f"Saved {os.path.basename(target)}")
    elif len(batch_files) == 1:
        target = os.path.join(saving_folder, os.path.basename(batch_files[0]))
        with open(target, 'wb') as f:
            f.write(content)
        print(f"Saved {os.path.basename(target)}")
    else:
        raise ValueError("Unexpected answer from the server (not a ZIP archive) for a multi-file request")


def download(file_list, saving_folder="EilatDS", batch_size=6):
    import requests

    os.makedirs(saving_folder, exist_ok=True)
    for i in range(0, len(file_list), batch_size):
        batch_number = i // batch_size + 1
        # Only request the files that are still missing
        batch_files = [f for f in file_list[i:i + batch_size]
                       if not os.path.exists(os.path.join(saving_folder, os.path.basename(f)))]
        if not batch_files:
            print(f"Skipping batch {batch_number} (all files already downloaded)")
            continue

        url = BASE_URL + "".join(f"&files[]={name}" for name in batch_files)
        print(f"Downloading batch {batch_number} with {len(batch_files)} files")
        response = requests.get(url, timeout=600)
        response.raise_for_status()
        save_response(response.content, batch_files, saving_folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--file_list', default='../file_list.txt', help='Text file containing the download links')
    parser.add_argument('--output', default='EilatDS', help='Folder where the recordings are saved')
    parser.add_argument('--batch_size', type=int, default=6, help='Number of files requested at once')
    args = parser.parse_args()

    file_list = extract_filenames_from_file(args.file_list)
    if not file_list:
        parser.error(f"No .flac file name found in '{args.file_list}'")
    download(file_list, args.output, args.batch_size)


if __name__ == "__main__":
    main()
