import pyedflib
from pyedflib import EdfReader
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

reader = EdfReader("../../Sofia_recordings/Rec1.edf")

labels = reader.getSignalLabels()
#print("labels:", labels)


x = []
x.extend(range(0,5695600))
y = []
y = np.arange(-0.003, 0.003, 0.001)
print(y)


#### Getting Spo2 levels
spo2_index = labels.index("SpO2")
spo2_data = reader.readSignal(spo2_index)
spo2_data = list(filter(lambda x: x > 1, spo2_data))

plt.subplot(221)
plt.hist(spo2_data, facecolor='blue', alpha=0.5)
plt.subplot(222)
plt.plot(spo2_data)
plt.show()
####

exit(0)
rightLeg_index = labels.index("Right Leg")
rightLeg_data = reader.readSignal(rightLeg_index)
leftLeg_index = labels.index("Left Leg")
leftLeg_data = reader.readSignal(leftLeg_index)
# print(len(rightLeg_data))
# print(reader.file_duration)

plt.subplot(221)
plt.plot(rightLeg_data)
plt.title("Right Leg")
plt.subplot(222)
plt.plot(leftLeg_data)
plt.title("Left Leg")
plt.show()



#### Plotting Es
E1_index = labels.index("E1")
E2_index = labels.index("E2")
E3_index = labels.index("E3")
E4_index = labels.index("E4")
E1_data = reader.readSignal(E1_index)
E2_data = reader.readSignal(E2_index)
E3_data = reader.readSignal(E3_index)
E4_data = reader.readSignal(E4_index)
#print(dir(E1_index)) #### METHODS
#print(len(E1_data))

# plt.subplot(221)
# plt.plot(E1_data)
# plt.grid(True)
# plt.subplot(222)
# plt.plot(E2_data)
# plt.grid(True)
# plt.subplot(223)
# plt.plot(E3_data)
# plt.grid(True)
# plt.subplot(224)
# plt.plot(E4_data)
# plt.grid(True)
#plt.show()
####
