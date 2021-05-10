from tokenizers import Tokenizer
import pandas as pd
from pathlib import Path
import os
import kaggle
import torch
import numpy as np


class GetKaggle:

    def __init__(self,kaggle_directory, target_col, tokenizer_conf='Tokenizers/tokenizer-oscar.json',
                 features_cols=False, path="data/kaggle"):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.tokenizer = Tokenizer.from_file(tokenizer_conf)
        self.tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")
        self.tokenizer.enable_truncation(400)

        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(kaggle_directory, path=path, unzip=True)
        file = ' '.join([str(elem) for elem in os.listdir(path) if elem.endswith('.csv')])
        if len(os.listdir(path)) > 1:
            print(file)
            raise Exception("more than one file found")
        data = pd.read_csv(path+'/'+file)
        y = data.iloc[:,target_col].apply(lambda x: round(float(x.replace(',', '.'))))
        if features_cols:
            X = data[features_cols]
        X = data.iloc[:,:target_col].apply(''.join, axis=1)
        # max_length = int(X.apply(len).mean())
        # X = X.apply(lambda x: x[:max_length] if len(x)>max_length else x+(int(max_length-len(x)))*" [PAD] ")
        # print('Minimum length ', X.apply(len).min(), ' Maximum length ', X.apply(len).max(),
        #       '\nStandard deviation', X.apply(len).std(), ' Mean', X.apply(len).mean(),
        #       '\nInput_sequence', max_length)
        #X = X.apply(self.tokenizer.encode)

        # print('tokenized sentence\n', X[1].tokens, '\ndecoded text\n',
        #       self.tokenizer.decode(X[1].ids), '\nLabel\n', y[1])
        # print(X.head())

        self.features = X
        self.labels = y

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        current_sample = self.tokenizer.encode(self.features.iloc[idx]).ids
        if len(current_sample) < 200:
            print(len(current_sample))
            current_sample =current_sample.append([[3]*(100-len(current_sample))])
        elif len(current_sample) > 200:
            print(len(current_sample))
            current_sample = current_sample[:201]
        else:
            current_sample = current_sample
        current_target = self.labels.iloc[idx]
        print(current_sample)
        return {
            "x":torch.tensor(current_sample,dtype=torch.float),
            "y":torch.tensor(current_target,dtype=torch.long)
        }


url = 'mustfkeskin/turkish-movie-sentiment-analysis-dataset'
ds = GetKaggle(url, 2)
print('the length of our data \n',len(ds))
