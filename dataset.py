import urllib.request
import ssl
import zipfile
import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from utils import tokenizer
from config import cfg
from main import new_config

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. skipping donwload and extraction")
        return
    
    ssl_context = ssl._create_unverified_context()

    with urllib.request.urlopen(url, context=ssl_context) as res:
        with open(zip_path, "wb") as f:
            f.write(res.read())


    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extracted_path)

    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloadedf and saved as {data_file_path}")

download_and_unzip_data(url=url, zip_path=zip_path, extracted_path=extracted_path, data_file_path=data_file_path)

df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])

def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=28)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df

balanced_df = create_balanced_dataset(df)
print(f"dataset balanced.{balanced_df["Label"].value_counts()}")

balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=28).reset_index(drop=True) # shuffle
    train_end = int(len(df) * train_frac)
    val_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    return train_df, val_df, test_df

train_df, val_df, test_df = random_split(balanced_df, 0.7, 0.1)

print("Len of train_df", len(train_df))
print("Len of val_df", len(val_df))
print("Len of test_df", len(test_df))

train_df.to_csv("train.csv", index=None)
val_df.to_csv("val.csv", index=None)
test_df.to_csv("test.csv", index=None)

class Dataset(Dataset):
    def __init__(self, file, tokenizer, max_length=None, pad_token_id=50256):
        super().__init__()

        self.data = pd.read_csv(file)

        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_length()
        else:
            self.max_length = max_length

            self.encoded_texts = [
                encoded_text[:self.max_length] for encoded_text in self.encoded_texts
            ]

        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
    
    def __len__(self):
        return len(self.data)
    
    def _longest_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length



train_dataset = Dataset(file="train.csv", max_length=1024, pad_token_id=50256, tokenizer=tokenizer)
val_dataset = Dataset(file="val.csv", tokenizer=tokenizer, max_length=1024)
test_dataset = Dataset(file="test.csv", tokenizer=tokenizer, max_length=1024)

assert (train_dataset.max_length <= new_config["context_length"]), (
    f"dataset max_length {train_dataset.max_length} exceeded model context_length {new_config["context_length"]}"
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=new_config["batch_size"],
    shuffle=True,
    num_workers=new_config["num_workers"],
    drop_last=True
)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=new_config["batch_size"],
    shuffle=False,
    num_workers=new_config["num_workers"],
    drop_last=False
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=new_config["batch_size"],
    shuffle=False,
    num_workers=new_config["num_workers"],
    drop_last=False
)


 

