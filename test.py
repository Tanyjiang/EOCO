from util import EoCNet
import torch
from model import Model
import time
numclass=1
batch_size=8
task_size=1
memory_size=150
epochs = 300
learning_rate=1e-5

start_time_total = time.time()
feature_extractor=Model()
model=EoCNet(numclass,feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate)

for i in range(6):
    model.test()
end_time_total = time.time()
print('total time cost:{:.2f} hours'.format((end_time_total-start_time_total)/3600))