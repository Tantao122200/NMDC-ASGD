import pandas as pd
import os

readme = []
with open('./readme.txt') as metafile:
    records = metafile.readlines()
    for i in range(len(records)):
        fields = records[i].strip().split(" ")
        readme.append(float(fields[-1]))

readme.sort()

data_dir = "./test_data"
file_list = os.listdir(data_dir)
del_list = []
for file in file_list:
    myfile = os.path.join(data_dir, file)
    try:
        train_result = pd.read_csv(myfile, header=None, usecols=[3]).values.T
        if train_result[0][-1] < 0.83:
            del_list.append(myfile)
        else:
            print(myfile + " " + str(train_result[0][-1]))
    except Exception as e:
        print(e)
        print(myfile)


for file in del_list:
    os.remove(file)