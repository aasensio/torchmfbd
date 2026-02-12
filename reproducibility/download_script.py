import requests
import zipfile
import os
import sys
from tqdm import tqdm # Optional: for a progress bar

def download_and_extract_nextcloud_share(share_url, output_dir="."):
    """
    Downloads the content of a Nextcloud/Owncloud public share link
    (as a zip file) and extracts it.

    Args:
        share_url (str): The public share URL.
        output_dir (str): The directory where the contents should be extracted.
                           Defaults to the current directory.
    """
    # Construct the download URL
    if not share_url.endswith('/'):
        share_url += '/'
    download_url = share_url + 'download'
    
    # --- Derive filenames ---
    # Try to get a sensible filename from the last part of the share URL path
    # or default to 'downloaded_content'
    try:
        base_name = share_url.split('/')[-3] # Usually the share token part
        if not base_name or base_name == 's': # Handle potential weird URLs
             base_name = share_url.split('/')[-2]
        if not base_name or base_name == 's': # Fallback further
            base_name = "downloaded_content"
    except IndexError:
        base_name = "downloaded_content"
        
    zip_filename = os.path.join(output_dir, f"{base_name}.zip")
    # The folder name inside the zip is often the original shared folder name
    # We don't know it for sure beforehand, but often matches the URL component
    # We will extract *everything* into the output_dir anyway.
    
    print(f"Attempting to download from: {download_url}")
    
    try:
        # Make the request with streaming enabled for large files
        response = requests.get(download_url, stream=True, timeout=60) # Added timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # Get total file size for progress bar (optional)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte

        print(f"Downloading to: {zip_filename}")
        
        # Use tqdm for progress bar if installed
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True) if total_size > 0 else None
        
        # Save the downloaded content to a zip file
        with open(zip_filename, 'wb') as f:
            for data in response.iter_content(block_size):
                if progress_bar:
                    progress_bar.update(len(data))
                f.write(data)
                
        if progress_bar:
            progress_bar.close()
            if total_size != 0 and progress_bar.n != total_size:
                 print("Error: Download size mismatch!", file=sys.stderr)


        print(f"\nDownload complete: {zip_filename}")

        # --- Extract the ZIP file ---
        print(f"Extracting '{zip_filename}' to '{output_dir}'...")
        try:
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            print(f"Extraction complete. Contents are in '{output_dir}'.")
            
            # --- Optional: Clean up the zip file ---
            try:
                cleanup = input(f"Do you want to delete the downloaded zip file '{zip_filename}'? (y/N): ")
                if cleanup.lower() == 'y':
                    os.remove(zip_filename)
                    print(f"Deleted '{zip_filename}'.")
            except Exception as e:
                print(f"Could not delete zip file: {e}", file=sys.stderr)
                
        except zipfile.BadZipFile:
            print(f"Error: Failed to extract '{zip_filename}'. It might be corrupted or not a valid zip file.", file=sys.stderr)
        except Exception as e:
            print(f"An error occurred during extraction: {e}", file=sys.stderr)

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}", file=sys.stderr)
        # Clean up potentially incomplete zip file on error
        if os.path.exists(zip_filename):
             try:
                 os.remove(zip_filename)
                 print(f"Cleaned up incomplete file: {zip_filename}")
             except OSError as oe:
                 print(f"Could not remove incomplete file {zip_filename}: {oe}", file=sys.stderr)
                 
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)


# --- Usage ---
share_link = "https://cloud.iac.es/index.php/s/EqMGsqBeyfq6Bnr"
destination_folder = "downloaded_obs" # Specify the folder to extract into

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
    print(f"Created directory: {destination_folder}")

download_and_extract_nextcloud_share(share_link, destination_folder)