import os
import re
import json
from tqdm import tqdm
from datasets import load_dataset
import pandas as pd

def clean_text(text):
    """
    Remove non-ASCII characters
    Replace multiple spaces with single space
    Remove unwanted characters
    """
    text = text.encode('ascii',errors='ignore').decode()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?;\'"()-]', '', text)
    return text.strip()

def preprocess_reddit_prompts(raw_file,processed_file):

    df = pd.read_csv(raw_file)
    prompts= df['prompt'].tolist()

    with open(processed_file, 'w',encoding='utf-8') as outfile:
        for prompt in tqdm(prompts, desc="Processing Reddit prompts"):
            cleaned = clean_text(prompt)
            outfile.write(cleaned + '\n')

    print(f"Reddit prompts preprocessed and saved to {processed_file}")

def preprocess_creative_prompts(raw_file,processed_file):
    df= pd.read_csv(raw_file)
    prompts= df['prompt'].tolist()

    with open(processed_file,'w',encoding='utf-8') as outfile:
        for prompt in tqdm(prompts, desc="Processing Creative prompts"):
            cleaned= clean_text(prompt)
            outfile.write(cleaned + '\n')

    print(f"Creative prompts preprocessed and saved to {processed_file}")

def preprocess_WikiText():
    dataset= load_dataset("wikitext", "wikitext-103-raw-v1")
    texts= dataset['train']['text']
    processed_file= './data/processed/wikitext_processed.txt'
    with open(processed_file, 'w', encoding='utf-8') as outfile:
        for text in tqdm(texts, desc="Processing WikiText"):
            cleaned = clean_text(text)
            outfile.write(cleaned + '\n')

    print(f"WikiText preprocessed and saved to {processed_file}")

if __name__ == "__main__":
    raw_reddit_file = './data/raw/reddit_writing_prompts.csv'  
    processed_reddit_file= './data/processed/reddit_prompts_processed.txt'
    preprocess_reddit_prompts(raw_reddit_file, processed_reddit_file)

    raw_creative_file = './data/raw/creative_writing_prompts.csv' 
    processed_creative_file= './data/processed/creative_prompts_processed.txt'
    preprocess_creative_prompts(raw_creative_file, processed_creative_file)
    
    raw_wikihow_file= ''
    processed_wikihow_file= ''
    preprocess_WikiText()
