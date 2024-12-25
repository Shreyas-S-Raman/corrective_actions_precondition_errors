import json
import sys
import os

from evaluate_online import evaluate_script
import pandas as pd


log_df = pd.read_csv('file.csv')
# column of VH scripts -- 'parsed_text'

for row in range(0, len(log_df)):
    # get the element in row
    script = log_df['parsed_text'][row]



