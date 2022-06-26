# %%
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer


# %%
def create_data_generator():
    import json
    import os
    from pandas.io.parsers.readers import read_csv

    # num data = 437948 + 12272
    newspaper_dir = 'bangla-newspaper'
    data_dirs = [dir_name for dir_name in os.listdir('../data') if dir_name != newspaper_dir]
    newspaper_df = read_csv('../data/bangla-newspaper/data.csv')

    yield from [title for title in newspaper_df['title'] if type(title) is str]
    yield from [content for content in newspaper_df['content'] if type(content) is str]

    for data_dir in data_dirs:
        data_dir_path = os.path.join('../data', data_dir)
        assert os.path.exists(data_dir_path), data_dir_path
        for dir_name, _, files in os.walk(data_dir_path):
            for filename in files:
                assert filename.endswith('.json'), os.path.join(dir_name, filename)

                filepath = os.path.join(dir_name, filename)
                with open(filepath, 'r', encoding='utf-8') as json_file:
                    data_obj = json.load(json_file)
                    yield data_obj['title']
                    yield data_obj['author']
                    yield data_obj['content']


# %%
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = WordPieceTrainer(vocab_size=32000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train_from_iterator(create_data_generator(), trainer, length=437948 + 12272)
# %%

tokenizer.save('../data/word-piece.json')
# %%
