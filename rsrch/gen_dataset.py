import json
from random import uniform, randint


dataset =[]

def gen_dataset():
    for i in range(100):
        sample = [[], []]
        for j in range(3):
            sample[0].append(randint(0, 10))
            sample[1].append(uniform(0, 1))
        dataset.append(sample)


def write():
    with open("test_dataset.json", "w") as f:
        json.dump(dataset, f)

def read():
    with open("test_dataset.json", "r") as f:
        data = json.load(f)
    print(data[0])

if __name__ == '__main__':
    gen_dataset()
    write()
    read()