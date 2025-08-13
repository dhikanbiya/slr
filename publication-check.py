import pandas as pd
from ollama import chat
from ollama import ChatResponse
from pathlib import Path



# prompt = '''"
# You are a classification bot. Your only function is to check a publication name and return a single, unadorned output based on a set of strict rules. You will not provide any explanations, justifications, or additional text.
# Instructions:
# Check the publication name provided.Determine if it is a journal or a conference.If it's a journal: Find its Scimago Journal Rank (SJR) quartile. Your output must be Q1, Q2, Q3, or Q4.
# If a Q ranking is not available: Find its general reputation. Your output must be YES (for reputable) or NO (for not reputable)
# If it's a conference: Find its general reputation. Your output must be YES (for highly reputable) or NO (for not reputable).
# If the publication cannot be found: Your output must be N/A
# Final Output: Provide only the final result (Q1, Q2, Q3, Q4, YES, NO, or N/A). Do not add any other words, sentences, or punctuation.
# Example Input: "Journal of Applied Physics"
# Example Output: "Q1"
# Example Input: "NeurIPS"
# Example Output: "YES"
# Example Input: "Weekly Research Gazette"
# Example Output: "N/A"
# "''' 



prompt = '''"Check the publication name, whether it is a journal or conference.If it is a journal, find its Scimago Journal Rank (SJR) quartile. Provide its Q ranking as Q1, Q2, Q3, or Q4.f the Scimago Journal Rank is unavailable, or if the journal is not indexed on Scimago, attempt to find its reputation from other sources. If it is a reputable journal, answer with "YES". If it is not, answer with "NO".If it is a conference, answer with "YES" if it is highly reputable and "NO" if it is not.If the publication name cannot be found or is not a known journal or conference, answer with "N/A".No explanation or additional text is needed; just the direct answer.Output format:Journals: Q1/Q2/Q3/Q4, YES, NO, or N/A
.Conferences: YES, NO, or N/A"''' 

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
       verdict= ollama_response(row['Publisher'])
       reputable.append(verdict)
    df['reputable']=reputable
    return df

def main(csv_file, llm_model):
    chunksize = 1000
    chunklist = []

    for chunk in pd.read_csv(csv_file, chunksize=chunksize):
        chunklist.append(chunk)    
    df = pd.concat(chunklist, ignore_index=True)

     
    nan_counts = df.isnull().sum()
    print(nan_counts)
    print(df[df['Publisher'].isnull()])
    df = df.dropna(subset=['Publisher'])

    df = df.head(10)
    df = add_classification(df)
    filename = Path(csv_file).stem
    df.to_csv(f'results/reputable_{filename}_{llm_model}_result.csv', index=False)
    print("saved")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='filter papers based on vulnerability detection relevance.')
    parser.add_argument('csv_file', type=str, help='Path to the input CSV file containing paper data.')
    parser.add_argument('llm_model', type=str, default='llama4:latest', help='Model to use for Ollama chat.')
    args = parser.parse_args()

    main(args.csv_file, args.llm_model)