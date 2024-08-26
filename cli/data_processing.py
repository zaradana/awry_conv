#%%
from convokit import Corpus, download
import pandas as pd

def construct_dataset(corpus_name):
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
            if not utterances.iloc[i].get("meta.is_section_header", False):
                texts.append(utterances.iloc[i].text)
            if len(texts) == 2:
                dataset.append({
                    "text": texts[0],
                    "reply": texts[1],
                    "goes_awry": 1 if conversation.meta[label] else 0,
                    "conversation_id": conv_id,
                    "corpus": corpus_name
                })
                break
    return dataset


wiki_dataset = construct_dataset("conversations-gone-awry-corpus")  # WIKI
reddit_dataset = construct_dataset("conversations-gone-awry-cmv-corpus")  # subreddit ChangeMyView

dataset = pd.concat([pd.DataFrame(wiki_dataset), pd.DataFrame(reddit_dataset)])

pd.DataFrame(dataset).to_csv("data/conversations-gone-awry.csv", index=False)

