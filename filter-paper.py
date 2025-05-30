import pandas as pd
from ollama import chat
from ollama import ChatResponse

def ollama_response(abstract):
    prompt  = "I want you to check my paragraph whether is related to vulnerability detection or not. Answer only with YES or NO. Yes if its related and No if its not related"
    response: ChatResponse = chat(model='llama4:latest',messages=[
        {"role": "user", "content": prompt},
        {"role": "user", "content": abstract}
        ])
    
    return response.message.content


def add_classification(df):
    df = df[['Authors','Abstract','Title','Year']].copy()

    related = []
    
    for index,row in df.iterrows():
       verdict= ollama_response(row['Abstract'])
       related.append(verdict)
    df['Related']=related
    return df

def main(csv_file):
    chunksize = 1000
    chunklist = []

    for chunk in pd.read_csv(csv_file, chunksize=chunksize):
        chunklist.append(chunk)    
    df = pd.concat(chunklist, ignore_index=True)
    
    df = add_classification(df)
    df.to_csv('result.csv', index=False)
    print("saved")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='filter papers based on vulnerability detection relevance.')
    parser.add_argument('csv_file', type=str, help='Path to the input CSV file containing paper data.')
    args = parser.parse_args()
    
    main(args.csv_file)