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

test_data = parse_directory(directory_path="./SygnalsWithoutNoise")
x_test, y_test = splitSamples(test_data)

batch_size = 32
test_dataset = DynamicNMRDataset(*x_test, y=y_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

criterion = nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Используемое устройство:", device)

model.load_state_dict(torch.load('model_weights_100.pth', weights_only=True))
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

# [[0.25242278 0.17025308]
#  [0.3621249  0.15095504]
#  [0.54478914 0.17845103]
#  [0.30923316 0.1957617 ]
#  [0.11337855 0.20916207]
#  [0.27910006 0.18937339]
#  [0.10649697 0.20878705]
#  [0.01613051 0.23031151]
#  [0.15333009 0.15961665]
#  [0.35641652 0.15912607]
#  [0.5171688  0.15341431]
#  [0.44746524 0.16249561]
#  [0.69999236 0.1613776 ]
#  [0.99661785 0.16160746]
#  [0.93276054 0.16709314]
#  [0.8612164  0.16546202]]
# [[0.325 0.311]
#  [0.325 0.061]
#  [0.675 0.9  ]
#  [0.525 0.97 ]
#  [0.325 0.93 ]
#  [0.475 0.73 ]
#  [0.325 0.93 ]
#  [0.125 0.93 ]
#  [0.025 0.061]
#  [0.375 0.161]
#  [0.525 0.161]
#  [0.525 0.311]
#  [0.725 0.361]
#  [0.975 0.361]
#  [0.925 1.   ]
#  [0.875 0.9  ]]
# 0.14223264157772064


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