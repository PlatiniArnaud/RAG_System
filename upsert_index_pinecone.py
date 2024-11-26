import subprocess

# Install packages

subprocess.check_call(['pip', 'install', '-q', 'pinecone-client'])
subprocess.check_call(['pip', 'install', '-q', 'python-dotenv'])

# Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pinecone import Pinecone
from dotenv import load_dotenv
import ast, json


# Get the current working directory
path = os.getcwd()

# Create data directories
os.makedirs(path + '/Extracted_Images', exist_ok=True)
os.makedirs(path + '/indexDf', exist_ok=True)

# Set the data directories
dataDir = path + '/Data/'
imgSaveDir = path + '/Extracted_Images/'
indexSaveDir = path + '/indexDf'



load_dotenv('my_secrets.env')
pinecone_key = os.getenv('PINECONE_KEY')
pc = Pinecone(api_key = pinecone_key)
index = pc.Index('health-system')

index.describe_index_stats()

index_df = pd.read_csv(f'{indexSaveDir}/index_df.csv')

def prepare_DF(df):
  import json,ast
  try: df=df.drop('Unnamed: 0',axis=1)
  except: print('Unnamed Not Found')
  df['embeddings']=df['embeddings'].apply(lambda x: np.array([float(i) for i in x.replace("[",'').replace("]",'').split(',')]))
  df['metadata']=df['metadata'].apply(lambda x: ast.literal_eval(x))
  return df

index_df = prepare_DF(index_df)

index_df.rename(columns={'ids': 'id', 'embeddings':'values'},inplace=True)

def chunker(seq, size):
  'Yields a series of slices of the original iterable, up to the limit of what size is.'
  for pos in range(0, len(seq), size):
      yield seq.iloc[pos:pos + size]

def convert_data(chunk):
  'Converts a pandas dataframe to be a simple list of tuples, formatted how the `upsert()` method in the Pinecone Python client expects.'
  data = []
  for i in chunk.to_dict('records'):
      data.append(i)
  return data

for chunk in chunker(index_df, 200):
    index.upsert(vectors=convert_data(chunk))

index.describe_index_stats()



