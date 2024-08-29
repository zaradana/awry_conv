import argparse
from convokit import Corpus, download
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def process_data(corpus_name):
    dataset = []
    corpus = Corpus(filename=download(corpus_name))
    if corpus_name == "conversations-gone-awry-corpus":
        label = "conversation_has_personal_attack"
    elif corpus_name == "conversations-gone-awry-cmv-corpus":
        label = "has_removed_comment"
    else:
        raise ValueError("Invalid corpus name")
    
    for conv_id in corpus.get_conversation_ids():
        conversation = corpus.get_conversation(conv_id)
        utterances = conversation.get_utterances_dataframe().sort_values(by="timestamp")
        texts = []
        for i in range(0, len(utterances)):
            if not utterances.iloc[i].get("meta.is_section_header", False) and utterances.iloc[i].text.strip() != '':
                texts.append(utterances.iloc[i].text)
            if len(texts) >= 2:
                dataset.append({
                    "text": texts[0],
                    "reply": texts[1],
                    "goes_awry": 1 if conversation.meta[label] else 0,
                    "conversation_id": conv_id,
                    "corpus": corpus_name
                })
                break
    return dataset

def construct_train_eval_dataset(data_dir):
    wiki_dataset = process_data("conversations-gone-awry-corpus")  # WIKI
    reddit_dataset = process_data("conversations-gone-awry-cmv-corpus")  # subreddit ChangeMyView

    dataset = pd.concat([pd.DataFrame(wiki_dataset), pd.DataFrame(reddit_dataset)])
    print(f"number of rows: {len(dataset)}")    

    os.makedirs(data_dir, exist_ok=True)
    pd.DataFrame(dataset).to_csv(os.path.join(data_dir, "conversations-gone-awry.csv"), index=False)

    train_df, eval_df = train_test_split(dataset, test_size=0.2, stratify=dataset['goes_awry'], random_state=42)
    train_df.to_csv(os.path.join(data_dir, "conversations-gone-awry-train.csv"), index=False)
    eval_df.to_csv(os.path.join(data_dir, "conversations-gone-awry-eval.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process conversation data and save to CSV files.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to save the processed data")
    args = parser.parse_args()
    
    construct_train_eval_dataset(args.data_dir)


