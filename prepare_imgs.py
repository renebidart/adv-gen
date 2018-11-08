# prepare art dataset: https://www.kaggle.com/c/painter-by-numbers/data

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument('--files_dict_loc', type=str)
parser.add_argument('--IM_SIZE', default=64, type=int)
args = parser.parse_args()

def main(args):
	PATH = Path(args.files_dict_loc).parent

	with open(args.files_dict_loc, 'rb') as f:
	    files_dict = pickle.load(f)

	# del files_dict['train'] 

	for folder, files_df in files_dict.items():
	    for index, row in tqdm(files_df.iterrows(), total=files_df.shape[0]):
	        try:
	            loc = str(PATH) +'/train/'+ row['path']
	#             for size in [64]:
	            img = Image.open(loc).convert('RGB').resize((args.IM_SIZE, args.IM_SIZE))
	            img.save(str(PATH) +'/'+str('train')+'_'+str(args.IM_SIZE)+'/'+ row['path'],"PNG", compress_level=2)
	        except Exception as e:
	            print(e)

if __name__ == '__main__':
    main(args)
