# Imports
import os
import torch
from torch.optim import Adam
from data import get_data, denorm
from models import Model
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import save_image

# Hyper params
batch_size = 32
num_epochs = 100
lr = 1e-8
ss_interval = 50

loss_dir = './loss'
model_state_dir = './models/model_state'
image_dir = './save/image'
train_content_dir = './data/train/content/'
train_style_dir = './data/train/style/'
test_content_dir = './data/test/content/'
test_style_dir = './data/test/style/'

loadmodel = None

def main():

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

    # start training
    loss_list = []
    for e in range(1, num_epochs + 1):
        print(f'Start {e} epoch')
        for i, (content, style) in tqdm(enumerate(train_loader, 1)):
            content = content.to(device)
            style = style.to(device)
            loss = model(content, style)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            x = round(iters/batch_size)
            print(f'{e}/{num_epochs} | steps: {i}/{x} | loss: {loss.item()}')

            if i % ss_interval == 0:
                content, style = next(test_iter)
                content = content.to(device)
                style = style.to(device)
                with torch.no_grad():
                    out = model.generate(content, style)
                content = denorm(content, device)
                style = denorm(style, device)
                out = denorm(out, device)
                res = torch.cat([content, style, out], dim=0)
                res = res.to('cpu')
                save_image(res, f'{image_dir}/{e}_epoch_{i}_iteration.png', nrow=batch_size)
        torch.save(model.state_dict(), f'{model_state_dir}/{e}_epoch.pth')
    plt.plot(range(len(loss_list)), loss_list)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('train loss')
    plt.savefig(f'{loss_dir}/train_loss.png')
    with open(f'{loss_dir}/loss_log.txt', 'w') as f:
        for l in loss_list:
            f.write(f'{l}\n')
    print(f'Loss saved in {loss_dir}')

if __name__ == '__main__':
    main()