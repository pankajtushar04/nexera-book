import os
import urllib.request
import zipfile
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
target_dir = os.path.expanduser("~/.insightface/models/buffalo_l")
zip_path = os.path.expanduser("~/.insightface/models/buffalo_l.zip")

os.makedirs(os.path.expanduser("~/.insightface/models"), exist_ok=True)

if not os.path.exists(target_dir):
    print("Downloading buffalo_l.zip from github...")
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete. Extracting...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    print("Extraction complete. Model is ready!")
    
    try:
        os.remove(zip_path)
    except:
        pass
else:
    print("Buffalo_l is already installed.")
