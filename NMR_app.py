import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from NMR_model import DynamicNMRRegressor, DynamicNMRDataset
from NMR_preproc_methods import parse_directory, splitSamples
import numpy as np

# check up
input_dims=[400, 60]
num_targets=2

# init model
model = DynamicNMRRegressor(input_dims, num_targets)

test_data = parse_directory(directory_path="./SygnalsNoise_005_001") #SygnalsWithoutNoise
x_test, y_test = splitSamples(test_data)

batch_size = 32
test_dataset = DynamicNMRDataset(*x_test, y=y_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

criterion = nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Используемое устройство:", device)

model.load_state_dict(torch.load('model_weights_100noise003.pth', weights_only=True)) #noise003
model.to(device)
model.eval() 

test_running_loss = 0.0
all_preds = []
all_targets = []

with torch.no_grad():
    for batch in test_dataloader:
        *x_batch, y_batch = batch
        x_batch = [x.to(device) for x in x_batch]
        y_batch = y_batch.to(device)
        
        outputs = model(*x_batch)
        loss = criterion(outputs, y_batch)
        test_running_loss += loss.item()
        
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())

# предсказания и цели
all_preds = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)
test_loss = test_running_loss / len(test_dataloader)

print(all_preds)
print(all_targets)
print(test_loss)

def getErrorsMaxMean(errors):
    print("МАКСИМАЛЬНОЕ ОТКЛОНЕНИЕ ОТ ИСТИНОГО ЗНАЧЕНИЯ =", max(errors))
    print("СРЕДНЕЕ ОТКЛОНЕНИЕ ОТ ИСТИНОГО ЗНАЧЕНИЯ =", np.sum(np.abs(errors)) / len(errors)) 
    
for i in range(num_targets):
    errors = all_preds[:, i] - all_targets[:, i]
    print("ДЛЯ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ", i)
    getErrorsMaxMean(errors)

# [[0.7603017  0.04667961]
#  [0.75721484 0.23085774]]
# [[0.8   0.015]
#  [0.8   0.265]]
# 0.00139395403675735

# [[0.28497803 0.29719105]
#  [0.2962841  0.07279351]]
# [[0.325 0.311]
#  [0.325 0.061]]
# 0.0006890330696478486

# [[0.80997837 0.06882986]
#  [0.7600721  0.07933292]
#  [0.04863715 0.23367113]
#  [0.07639858 0.23570764]
#  [0.05443848 0.25006574]
#  [0.04315719 0.23895165]
#  [0.12911844 0.19742769]
#  [0.2509784  0.1331088 ]
#  [0.719747   0.09323786]
#  [0.9016969  0.0783102 ]
#  [0.94941664 0.07879589]
#  [0.9563873  0.07805514]
#  [0.30897462 0.18350531]
#  [0.18430118 0.21587218]
#  [0.27485055 0.15827535]
#  [0.09101228 0.20342824]]
# [[0.8   0.015]
#  [0.8   0.265]
#  [0.1   0.765]
#  [0.25  0.965]
#  [0.15  1.   ]
#  [0.05  0.75 ]
#  [0.325 0.611]
#  [0.325 0.211]
#  [0.8   0.515]
#  [0.95  0.515]
#  [1.    0.765]
#  [1.    0.015]
#  [0.55  0.865]
#  [0.45  0.965]
#  [0.45  0.465]
#  [0.25  0.615]]
# 0.1334417313337326


# [[0.801755   0.08676932]
#  [0.7980396  0.18778458]
#  [0.44481447 0.86284816]
#  [0.6019489  0.92566335]
#  [0.5721447  1.0237103 ]
#  [0.42377684 0.8785967 ]
#  [0.49577716 0.60254514]
#  [0.33557054 0.22443753]
#  [0.8158191  0.29795253]
#  [0.9489389  0.20165384]
#  [0.96434885 0.19568336]
#  [1.0011643  0.16024601]
#  [0.7295406  0.6347929 ]
#  [0.7067381  0.77157074]
#  [0.5276409  0.42559853]
#  [0.44278714 0.6391094 ]]
# [[0.8   0.015]
#  [0.8   0.265]
#  [0.1   0.765]
#  [0.25  0.965]
#  [0.15  1.   ]
#  [0.05  0.75 ]
#  [0.325 0.611]
#  [0.325 0.211]
#  [0.8   0.515]
#  [0.95  0.515]
#  [1.    0.765]
#  [1.    0.015]
#  [0.55  0.865]
#  [0.45  0.965]
#  [0.45  0.465]
#  [0.25  0.615]]
# 0.04235859215259552