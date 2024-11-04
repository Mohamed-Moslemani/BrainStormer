import os
import re
import json
from tqdm import tqdm
from datasets import load_dataset
import pandas as pd

def clean_text(text):

    text= text.encode('ascii', errors='ignore').decode()
    text = re.sub(r'\s+', ' ', text)
    text= re.sub(r'[^\w\s.,!?;\'"()-]', '', text)
    return text.strip()

def preprocess_and_append(prompts,output_file, description):
    with open(output_file, 'a', encoding='utf-8') as outfile:
        for prompt in tqdm(prompts,desc=f"Processing {description}"):
            cleaned = clean_text(prompt)
            outfile.write(cleaned + '\n')
    print(f"{description} data appended to {output_file}")

def preprocess_reddit_stories(raw_file,output_file):
    df = pd.read_csv(raw_file)
    prompts = df['story'].tolist()  
    preprocess_and_append(prompts,output_file,"Reddit Stories")

def preprocess_wikihow(output_file):
    dataset = load_dataset("wikihow", "all", split="train")
    texts = dataset['text']
    preprocess_and_append(texts,output_file, "WikiHow")

def preprocess_openweb(output_file):
    dataset = load_dataset("openwebtext")
    texts = dataset['train']['text']
    preprocess_and_append(texts, output_file, "OpenWeb")

if __name__ == "__main__":
    combined_output_file = './data/processed/combined_processed.txt'
    if os.path.exists(combined_output_file):
        os.remove(combined_output_file)
    raw_reddit_stories_file = './data/raw/reddit_stories.csv'
    preprocess_reddit_stories(raw_reddit_stories_file, combined_output_file)
    
    preprocess_wikihow(combined_output_file)
    preprocess_openweb(combined_output_file)
