import pandas as pd
import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
stemmer = LancasterStemmer()
training_data = []
df = pd.read_csv('dataset.csv')

dtest = str(df[df["No"] == 5].Text).strip()
dtest = dtest[7:len(dtest)]
print(dtest)
if(dtest == "Biography"):
    print("one more step")

    # training_data.append({"class":df[x]["Genre"], "sentence":df[x]["Text"]})