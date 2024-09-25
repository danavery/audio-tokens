import json

train = set()

with open("output/bal_unbal_train_val_data_split.json", "r") as file:
    split = json.load(file)
    for filename in split["train"]:
        train.add(filename)
    for filename in split["validation"]:
        if filename in train:
            print(f"{filename} in both")
