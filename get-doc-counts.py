import sys
import os
import warnings
import glob
import csv
import pandas as pd

if not sys.warnoptions:
	warnings.simplefilter("ignore")
csv.field_size_limit(sys.maxsize)
data = sys.argv[1]
lang = sys.argv[2]
save_dir = 'doc-counts'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"doc-counts-{lang}.txt")
df_list = []
for filename in glob.glob(data):
	if '.tsv' in filename:
		df = pd.read_csv(filename, delimiter="\t")
	elif '.csv' in filename:
		df = pd.read_csv(filename, delimiter=",")
	else:
		full = []
		for line in open(filename, "r", encoding="utf-8"):
			new = []
			line=line.strip().split("\t")
			new.append(**line)
			full.append(new)
		df = pd.DataFrame(full,  columns = ['id', 'label', 'pred', 'token', 'score', 'probs'])
	df_list.append(df)
df_full = pd.concat(df_list)
del df_list
doc_counts = len(df_full.groupby('id'))
save_file = open(save_path, "w")
save_file.write(str(doc_counts))
save_file.close()
