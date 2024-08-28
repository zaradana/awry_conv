import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(filename)s->%(funcName)s():%(lineno)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    roc_auc = roc_auc_score(labels, pred.predictions[:, 1])  # Assuming binary classification with probability outputs
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
    parser.add_argument("--dataset-path", type=str, required=False, default="data/conversations-gone-awry-small.csv")
    parser.add_argument("--model-name", type=str, required=False, default="bert-base-cased")
    parser.add_argument("--output-dir", type=str, required=False, default="model")
    parser.add_argument("--col-label", type=str, required=False, default="goes_awry")
    parser.add_argument("--col-text-a", type=str, required=False, default="text")
    parser.add_argument("--col-text-b", type=str, required=False, default="reply")
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.dataset_path)

    # Ensure the label column is correctly named
    df = df.rename(columns={args.col_label: 'label', args.col_text_a: 'text_a', args.col_text_b: 'text_b'})
    logger.info(f"df size: {df.shape}")

    train_df, test_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)

    train_dataset = Dataset.from_pandas(train_df[['text_a', 'text_b', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['text_a', 'text_b', 'label']])

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    def tokenize_function(examples):
        return tokenizer(examples['text_a'], examples['text_b'], padding='max_length', truncation=True)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir+"_chpts",
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    logger.info(f"Eval f1 score is {metrics['eval_f1']}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
