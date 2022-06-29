# %%
import os
from datasets import Dataset
from pandas.io.parsers.readers import read_csv

# %%
newspaper_df = read_csv('../data/bangla-newspaper/data.csv')

newspaper_dataset = Dataset.from_pandas(newspaper_df)

# %%
poem_filepaths = []
for dir_name, _, files in os.walk('../data'):
    for filename in files:
        if filename.endswith('.json'):
            filepath = os.path.join(dir_name, filename)
            poem_filepaths.append(filepath)

poem_dataset = Dataset.from_json(poem_filepaths)

# %%
from datasets import concatenate_datasets

datasets = concatenate_datasets([newspaper_dataset, poem_dataset])
columns = ['author', 'title', 'content']
columns_to_remove = [col for col in datasets.column_names if col not in columns]
datasets = datasets.remove_columns(columns_to_remove)
datasets = datasets.train_test_split(test_size=10000)

print('Split sizes:', *map(len, datasets.values()))
# %% md

# Tokenize
# %%
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(tokenizer_file='../tokenizers/word-piece.json')


# %%
def map_row(row):
    return tokenizer(f'লেখো {row["author"]} {row["title"]}: {row["content"]}')


tokenized_dataset = datasets.map(map_row, batched=True, num_proc=12, remove_columns=columns)
# %%
from datasets import DatasetDict

BLOCK_SIZE = 512


def group_text(examples: DatasetDict):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE

    result = {
        k: [t[i: i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated_examples.items()
    }

    result['labels'] = result['input_ids'].copy()

    return result


# %%
lm_dataset = tokenized_dataset.map(
    group_text,
    batched=True,
    batch_size=1000,
    num_proc=12,
)
# %% md

# Train The Model
# %%
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_config(config)
# %%
from transformers import Trainer, TrainingArguments, IntervalStrategy

training_args = TrainingArguments(
    output_dir='gpt-genbn',
    evaluation_strategy=IntervalStrategy.EPOCH,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    log_level='info',

    load_best_model_at_end=True,
    metric_for_best_model='bleu',
    greater_is_better=True,
    auto_find_batch_size=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset['train'],
    eval_dataset=lm_dataset['test'],
)
# %%
trainer.train()
# %% md

# Evaluate
# %%
import math

eval_result = trainer.evaluate()
print('Perplexity:', math.exp(eval_result['eval_loss']))
# %%
