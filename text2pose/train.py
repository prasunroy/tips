# imports
import cv2
import datetime
import time
import torchvision
from torch.utils.tensorboard import SummaryWriter
from data.dataloader import create_dataloader
from text2pose_model import Text2PoseModel


# configurations
# -----------------------------------------------------------------------------
dataset_name = 'DF-PASS'

dataset_root = f'../datasets/{dataset_name}'
pose_heatmaps_dir = f'{dataset_root}/gaussian_heatmaps'
text_encoding_data = f'{dataset_root}/encodings.csv'
img_list_train = f'{dataset_root}/train_img_list.csv'
img_list_test = f'{dataset_root}/test_img_list.csv'

gpu_ids = [0]

noise_dim = 128
embed_dim = 84
heatmap_channels = 18
gradient_penalty = True

batch_size_train = 32
batch_size_test = 8
n_epoch = 1000
out_freq = 500
d_iters = 5

ckpt_id = None
ckpt_dir = None

run_info = f'[gp={gradient_penalty}][d_iters={d_iters}]'
out_path = '../output/text2pose'
# -----------------------------------------------------------------------------


# create timestamp and infostamp
timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
infostamp = f'_{run_info.strip()}' if run_info.strip() else ''

# create tensorboard logger
logger = SummaryWriter(f'{out_path}/runs/{timestamp}{infostamp}')

# create transforms
text_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])
pose_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

# create dataloaders
train_dataloader = create_dataloader(img_list_train, text_encoding_data, pose_heatmaps_dir,
                                     text_transform, pose_transform, batch_size_train, shuffle=True)
test_dataloader = create_dataloader(img_list_test, text_encoding_data, pose_heatmaps_dir,
                                    text_transform, pose_transform, batch_size_test, shuffle=False)

# create fixed batch for testing
fixed_test_batch = next(iter(test_dataloader))

# create model
model = Text2PoseModel(gpu_ids, noise_dim, embed_dim, heatmap_channels, gradient_penalty)
model.print_networks(verbose=False)

# load pretrained weights into model
if ckpt_id and ckpt_dir:
    model.load_networks(ckpt_dir, ckpt_id, verbose=True)

# train model
n_batch = len(train_dataloader)
w_batch = len(str(n_batch))
w_epoch = len(str(n_epoch))
n_iters = 0

for epoch in range(n_epoch):
    for batch, data in enumerate(train_dataloader):
        time_0 = time.time()
        model.set_inputs(data)
        model.optimize_parameters(d_iters=d_iters)
        losses = model.get_losses()
        loss_G = losses['lossG']
        loss_D = losses['lossD']
        time_1 = time.time()
        
        print(f'[TRAIN] Epoch: {epoch+1:{w_epoch}d}/{n_epoch} | Batch: {batch+1:{w_batch}d}/{n_batch} |',
              f'LossG: {loss_G:7.4f} | LossD: {loss_D:7.4f} | Time: {round(time_1-time_0, 2):.2f} sec |')
        
        if (n_iters % out_freq == 0) or (batch+1 == n_batch and epoch+1 == n_epoch):
            model.save_networks(f'{out_path}/ckpt/{timestamp}{infostamp}', n_iters, verbose=True)
            for loss_name, loss in losses.items():
                loss_group = 'LossG' if loss_name.startswith('lossG') else 'LossD'
                logger.add_scalar(f'{loss_group}/{loss_name}', loss, n_iters)
            model.set_inputs(fixed_test_batch)
            visuals = model.compute_visuals(padding=4)
            # logger.add_image(f'Iteration_{n_iters}', visuals, n_iters)
            cv2.imwrite(f'{out_path}/runs/{timestamp}{infostamp}/iteration_{n_iters}.png', visuals)
        
        n_iters += 1
