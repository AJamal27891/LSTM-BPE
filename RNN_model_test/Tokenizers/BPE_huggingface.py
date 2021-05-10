import os 
data=[]
import re

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)


for F in os.listdir('data'):
    if F.endswith('.txt'):
        with open(f'data/{F}') as corpus:
            for l in corpus:
                data.append(deEmojify(l))
            print("__Finished reading files___")

import numpy as np 
train = round(.9 * len(data))
valid = train+round(.07*len(data))
test = len(data)
print("train size ",train,"\nValidation size ",valid-train,"\nTest size ",-train-valid+test)

with open("oscar/oscar.train.raw", "a") as fp:
  for l in data[:train]:
   fp.writelines(l)
with open("oscar/oscar.test.raw", "a") as fp:
  for l in data[valid:test]:
   fp.writelines(l)
with open("oscar/oscar.valid.raw", "a") as fp:
  for l in data[train:valid]:
   fp.writelines(l)

lengths = [len(x.split())for x in data]
print('total count of tokens',sum(lengths ),'\nMean',np.mean(lengths),'\nMin',np.min(lengths),'\nMax',np.max(lengths),
      '\nMedian',np.median(lengths),'\nStandard deviation',np.std(lengths))
from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
from tokenizers.trainers import BpeTrainer

trainer = BpeTrainer(special_tokens=["[UNK]","[CLS]", "[SEP]", "[PAD]", "[MASK]"],vocab_size =130000,min_frequency=2)
from tokenizers.pre_tokenizers import Whitespace

tokenizer.pre_tokenizer = Whitespace()

files = [f"oscar/oscar.{split}.raw" for split in ["test", "train", "valid"]]

tokenizer.train(files, trainer)
tokenizer.save("tokenizer-oscar.json")
tokenizer = Tokenizer.from_file("tokenizer-oscar.json")

inputtext = '''Herkese merhabalar, ilk makalemin yayım heyecanını sizlerle de paylaşmak istedim.
Merak edenler sf 177'de bulabilirler. Üşenenler için: ortaokul öğrencileri ile siber zorbalık farkındalığına ilişkin yaptığımız saha çalışması bulgularını ve literatür taramamızı içeren bir makale.
🥰
'''
output = tokenizer.encode(inputtext)
print('The input text after tokenization \n',output.tokens)
print('The input ids \n',output.ids)
unkown_idx = output.tokens.index('[UNK]')
original_idx = output.offsets[unkown_idx]
print('What is the oringial for of my special token [UNK]\n',inputtext[original_idx[0]:original_idx[1]])


