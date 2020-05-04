from Loader import Loader, samplify
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

loader = Loader("../../Sofia_recordings", ["SpO2"], ["SpO2"])

#print(dir(loader.x_channels))
count = 0
# for x,y in loader.load():
#     count += 1
#     print("COUNT: ",count)
#     print("X:",x)
#     print("Y:",y)


for x_train, x_test in loader.load():
    if x_train == "":
        break
    x_train = samplify(x_train, 1)
    for i in x_train:
        print(f'{i}:\t{len(x_train[i]["data"])}\t{x_train[i]["sampling_frequency"]}')
print(x_train.keys())
x = []
x.extend(range(0,5695600))

print(len(x_train.values()))
# plt.plot(x_train.values(), x)
# plt.show()
