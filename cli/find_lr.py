import pandas as pd
from datasets import Dataset
from torch_lr_finder import LRFinder
from transformers import AdamW, BertForSequenceClassification, BertTokenizer
from utils.utils import tokenize_function   
import torch
import os

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=False, default="./data", help="Directory containing the dataset files")
    parser.add_argument("--model-name", type=str, required=False, default="bert-base-uncased", help="Model name")
    parser.add_argument("--train-dataset-path", type=str, required=False, default="conversations-gone-awry-train.csv", help="Path to the training dataset")
    parser.add_argument("--eval-dataset-path", type=str, required=False, default="conversations-gone-awry-eval.csv", help="Path to the evaluation dataset")
    parser.add_argument("--col-label", type=str, required=False, default="goes_awry", help="Column name for the label")
    parser.add_argument("--col-text-a", type=str, required=False, default="text", help="Column name for the text_a")
    parser.add_argument("--col-text-b", type=str, required=False, default="reply", help="Column name for the text_b")
    return parser.parse_args()

def custom_collate_fn(batch, device):
    input_ids = torch.stack([item['input_ids'] for item in batch]).to(device)
    attention_mask = torch.stack([item['attention_mask'] for item in batch]).to(device)
    labels = torch.tensor([item['label'] for item in batch]).to(device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}, labels
    
class BertLRFinder(LRFinder):
    def __init__(self, model, optimizer, criterion, device):
        super().__init__(model, optimizer, criterion, device)

    def _train_batch(self, batches, accumulation_steps=1, non_blocking_transfer = False):
        self.model.train()
        self.optimizer.zero_grad()
        inputs, labels = next(batches)
        outputs = self.model(**inputs)
        loss = self.criterion(outputs.logits, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _validate(self, batches, non_blocking_transfer = False):
        self.model.eval()
        with torch.no_grad():
            running_loss = 0
            for inputs, labels in batches:
                outputs = self.model(**inputs)
                loss = self.criterion(outputs.logits, labels)
                running_loss += loss.item() * len(labels)

        return running_loss / len(batches.dataset)

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset_path = os.path.join(args.data_dir, args.train_dataset_path)
    eval_dataset_path = os.path.join(args.data_dir, args.eval_dataset_path)
    train_df = pd.read_csv(train_dataset_path)
    eval_df = pd.read_csv(eval_dataset_path)

    train_df = train_df.rename(columns={args.col_label: 'label', args.col_text_a: 'text_a', args.col_text_b: 'text_b'})
    eval_df = eval_df.rename(columns={args.col_label: 'label', args.col_text_a: 'text_a', args.col_text_b: 'text_b'})

    train_dataset = Dataset.from_pandas(train_df[['text_a', 'text_b', 'label']])
    eval_dataset = Dataset.from_pandas(eval_df[['text_a', 'text_b', 'label']])

    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(device)
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    train_dataset = train_dataset.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
    
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=1e-7, weight_decay=1e-2)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: custom_collate_fn(x, device))
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: custom_collate_fn(x, device))

    lr_finder = BertLRFinder(model, optimizer, criterion, device=device)

    lr_finder.range_test(trainloader, val_loader=eval_loader, end_lr=100, num_iter=100, non_blocking_transfer = False)

    ax = lr_finder.plot(log_lr=False, suggest_lr=False)
    fig = ax.get_figure()  
    fig.savefig(os.path.join(args.data_dir, "lr_plot.png"))  
    lr_finder.reset()


if __name__ == "__main__":
    main()