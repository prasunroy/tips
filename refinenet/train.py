# imports
import datetime
import json
import os
import torch
from dataloader import create_dataloader
from refinenet import RefineNet


# configurations
# -----------------------------------------------------------------------------
dataset_name = 'DF-PASS'
data_root = f'../datasets/{dataset_name}'
keypoints_data_train =  f'{data_root}/train_img_keypoints.csv'
keypoints_data_test = f'{data_root}/test_img_keypoints.csv'
output_dir = f'../output/refinenet/{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
noise_range = (-5, 5)
batch_size = 128
num_epochs = 100
learning_rate = 1e-2
use_gpu = True
# -----------------------------------------------------------------------------


# create dataloaders
train_dataloader = create_dataloader(keypoints_data_train, noise_range, batch_size=batch_size, shuffle=True)
test_dataloader = create_dataloader(keypoints_data_test, noise_range, batch_size=batch_size, shuffle=False)


# create model
model = RefineNet(10, 10, bias=True)
if use_gpu and torch.cuda.is_available():
    model.cuda()


# create objective and optimizer
mse = torch.nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=learning_rate)


# create output directory
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)


w1 = len(str(num_epochs))
w2 = len(str(len(train_dataloader)))

history = {'train_loss': [], 'eval_loss': []}
best_loss = None

for i_epoch in range(num_epochs):
    # train
    model.train()
    total_loss = 0
    total_samples = 0
    for i_batch, data in enumerate(train_dataloader):
        x = data['x'] / 50.0
        y = data['y'] / 50.0
        if use_gpu and torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        pred = model(x)
        loss = mse(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += pred.size(0) * loss.item()
        total_samples += pred.size(0)
        train_loss = total_loss / total_samples
        print(f'\rEpoch: {i_epoch+1:{w1}d}/{num_epochs} | Batch: {i_batch+1:{w2}d}/{len(train_dataloader)} | Loss: {train_loss:.4f}', end='')
    # eval
    model.eval()
    total_loss = 0
    total_samples = 0
    for i_batch, data in enumerate(test_dataloader):
        x = data['x'] / 50.0
        y = data['y'] / 50.0
        if use_gpu and torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        with torch.no_grad():
            pred = model(x)
        loss = mse(pred.detach().cpu(), y.detach().cpu())
        total_loss += pred.size(0) * loss.item()
        total_samples += pred.size(0)
    eval_loss = total_loss / total_samples
    print(f' | Validation Loss: {eval_loss:.4f}', end='')
    # save model
    if best_loss is None or eval_loss < best_loss:
        best_loss = eval_loss
        torch.save(model.state_dict(), f'{output_dir}/refinenet_best.pth')
        print(' | New best!')
    else:
        print('')
    torch.save(model.state_dict(), f'{output_dir}/refinenet_last.pth')
    # save history
    history['train_loss'].append(train_loss)
    history['eval_loss'].append(eval_loss)
    with open(f'{output_dir}/history.json', 'w') as fp:
        json.dump(history, fp)
