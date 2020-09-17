# Imports
import os
import torch
from torch.optim import Adam
from data import get_data
from models import Model

# Hyper params
batch_size = 32
num_epochs = 100
lr = 1e-8
ss_interval = 50

loss_dir = './loss'
model_state_dir = './save/model_state'
image_dir = './save/image'
train_content_dir = './data/train/content/'
train_style_dir = './data/train/style/'
test_content_dir = './data/test/content/'
test_style_dir = './data/test/style/'

loadmodel = None

# create directories
if not os.path.exists('./save'):
    os.mkdir('./save/')

if not os.path.exists(loss_dir):
    os.mkdir(loss_dir)
    os.mkdir(model_state_dir)
    os.mkdir(image_dir)

# Set GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# prepare dataset and dataLoader
train_loader, iters = get_data(train_content_dir, train_style_dir, batch_size, True)
test_loader, _ = get_data(test_content_dir, test_style_dir, batch_size, False)
test_iter = iter(test_loader)

# model
model = Model.to(device)
if loadmodel is not None:
    model.load_state_dict(torch.load(loadmodel))
optimizer = Adam(model.parameters(), lr=lr)
