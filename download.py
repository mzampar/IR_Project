
# --- MODULES ---

import sys
from beir.datasets.data_loader import GenericDataLoader
from beir import util

# --- INPUT ARGS ---

url, data_folder = sys.argv[1:]

# --- MAIN ---

data_path = util.download_and_unzip(url, data_folder)
