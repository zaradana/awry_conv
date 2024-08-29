import pandas as pd
import os
import torch 
from datasets import Dataset
from utils.utils import tokenize_function   
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(filename)s->%(funcName)s():%(lineno)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=num_training_steps)



def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    roc_auc = roc_auc_score(labels, pred.predictions[:, 1])  
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
    }

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=False, default="./data", help="Directory containing the dataset files")
    parser.add_argument("--train-dataset-path", type=str, required=False, default="conversations-gone-awry-train.csv", help="Path to the training dataset")
    parser.add_argument("--eval-dataset-path", type=str, required=False, default="conversations-gone-awry-eval.csv", help="Path to the evaluation dataset")
    parser.add_argument("--model-name", type=str, required=False, default="bert-base-cased", help="Model name")
    parser.add_argument("--output-dir", type=str, required=False, default="model", help="Output directory")
    parser.add_argument("--col-label", type=str, required=False, default="goes_awry", help="Column name for the label")
    parser.add_argument("--col-text-a", type=str, required=False, default="text", help="Column name for the text_a")
    parser.add_argument("--col-text-b", type=str, required=False, default="reply", help="Column name for the text_b")
    return parser.parse_args()

def main():
    args = parse_args()
    train_dataset_path = os.path.join(args.data_dir, args.train_dataset_path)
    eval_dataset_path = os.path.join(args.data_dir, args.eval_dataset_path)
    train_df = pd.read_csv(train_dataset_path)
    eval_df = pd.read_csv(eval_dataset_path)

    train_df = train_df.rename(columns={args.col_label: 'label', args.col_text_a: 'text_a', args.col_text_b: 'text_b'})
    eval_df = eval_df.rename(columns={args.col_label: 'label', args.col_text_a: 'text_a', args.col_text_b: 'text_b'})

    train_dataset = Dataset.from_pandas(train_df[['text_a', 'text_b', 'label']])
    eval_dataset = Dataset.from_pandas(eval_df[['text_a', 'text_b', 'label']])

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    train_dataset = train_dataset.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})

    training_args = TrainingArguments(
        output_dir=args.output_dir+"_chpts",
        evaluation_strategy='epoch',
        learning_rate=1e-3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=4,
        weight_decay=0.01,
    )


    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    logger.info(f"Eval f1 score is {metrics['eval_f1']}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()