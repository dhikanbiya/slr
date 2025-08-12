import pandas as pd
from ollama import chat
from ollama import ChatResponse
from pathlib import Path


prompt = '"determine if the conferences or the journal is highly reputable. Answer only with the Q ranking for the journal for example Q1, Q2, Q3 etc. and for the confereces that higly reputable answer with YES. If the journal or the conferences is not highly reputable simply answer with NO"' 

def ollama_response(abstract):    
    response: ChatResponse = chat(model='llama4:latest',messages=[
        {"role": "user", "content": prompt},
        {"role": "user", "content": abstract}
        ])
    
    return response.message.content


def add_classification(df):
    df = df.copy()
    reputable = []
    
    for index,row in df.iterrows():
       verdict= ollama_response(row['Source title'])
       reputable.append(verdict)
    df['reputable']=reputable
    return df

def main(csv_file, llm_model):
    chunksize = 1000
    chunklist = []

    for chunk in pd.read_csv(csv_file, chunksize=chunksize):
        chunklist.append(chunk)    
    df = pd.concat(chunklist, ignore_index=True)

    has_nan  =df.isnull().values.any()    
    nan_counts = df.isnull().sum()
    print(nan_counts)
    print(df[df['Source title'].isnull()]) 
    df = df.dropna(subset=['Source title'])

    # df = df.head(10)
    df = add_classification(df)
    filename = Path(csv_file).stem
    df.to_csv(f'results/reputation_check_{filename}_{llm_model}_result.csv', index=False)
    print("saved")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='filter papers based on vulnerability detection relevance.')
    parser.add_argument('csv_file', type=str, help='Path to the input CSV file containing paper data.')
    parser.add_argument('llm_model', type=str, default='llama4:latest', help='Model to use for Ollama chat.')
    args = parser.parse_args()

    main(args.csv_file, args.llm_model)