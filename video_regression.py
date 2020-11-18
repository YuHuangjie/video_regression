import os
import shutil
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import skvideo.datasets
import skvideo.io
from tqdm import tqdm
import configargparse

from model import *

def get_3d_mgrid(shape):
    pixel_coords = np.stack(np.mgrid[:shape[0], :shape[1], :shape[2]], axis=-1).astype(np.float32)

    # normalize pixel coords onto [-1, 1]
    pixel_coords[..., 0] = pixel_coords[..., 0] / max(shape[0] - 1, 1)
    pixel_coords[..., 1] = pixel_coords[..., 1] / max(shape[1] - 1, 1)
    pixel_coords[..., 2] = pixel_coords[..., 2] / max(shape[2] - 1, 1)
    pixel_coords -= 0.5
    pixel_coords *= 2.
    # flatten 
    pixel_coords = torch.Tensor(pixel_coords).view(-1, 3)

    return pixel_coords

class Video:
    def __init__(self, path_to_video):
        if 'npy' in path_to_video:
            self.vid = np.load(path_to_video)
        elif 'mp4' in path_to_video:
            self.vid = skvideo.io.vread(path_to_video).astype(np.single) / 255.

        self.shape = self.vid.shape[:-1]
        self.channels = self.vid.shape[-1]

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video):

        self.vid = video

        self.mgrid = get_3d_mgrid(self.vid.shape)
        data = (torch.from_numpy(self.vid.vid) - 0.5) * 2
        self.data = data.view(-1, self.vid.channels)
        self.N_samples = self.mgrid.shape[0]

    def __len__(self):
        return self.N_samples

    def __getitem__(self, idx):
        return self.mgrid[idx, :], self.data[idx, :]

def train(model, train_dataloader, lr, epochs, logdir, epochs_til_checkpoint=10, 
    steps_til_summary=100, val_dataloader=None):
    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    os.makedirs(logdir, exist_ok=True)

    checkpoints_dir = os.path.join(logdir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    summaries_dir = os.path.join(logdir, 'summaries')
    os.makedirs(summaries_dir, exist_ok=True)

    writer = SummaryWriter(summaries_dir, purge_step=0)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, f'model_epoch_{epoch:04}.pt'))
                np.savetxt(os.path.join(checkpoints_dir, f'train_losses_epoch_{epoch:04}.txt'),
                           np.array(train_losses))

                if val_dataloader is not None:
                    tqdm.write("Running partial validation set...")
                    model.eval()
                    with torch.no_grad():
                        val_losses = []
                        for (model_input, gt) in val_dataloader:
                            model_input, gt = model_input.cuda(), gt.cuda()
                            val_loss = model_loss2(model, model_input, gt)
                            val_losses.append(val_loss)
                            if len(val_losses) > 10:
                                break

                        writer.add_scalar("val_loss", torch.mean(torch.Tensor(val_losses)), total_steps)
                        tqdm.write(f"val_loss {torch.mean(torch.Tensor(val_losses))}")
                    model.train()

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()

                model_input = model_input.cuda()
                gt = gt.cuda()

                train_loss = model_loss2(model, model_input, gt)
                writer.add_scalar('train_loss', train_loss.item(), total_steps)
                train_losses.append(train_loss.item())

                optim.zero_grad()
                train_loss.backward()
                optim.step()

                pbar.update(1)

                # evaludate
                if not total_steps % steps_til_summary:
                    tqdm.write(f"Epoch {epoch}, Total loss {train_loss:.6}, iteration time {time.time()-start_time:.6}")

                total_steps += 1

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pt'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))

p = configargparse.ArgumentParser()

p.add_argument('--logdir', type=str, default='./logs/default', help='root for logging')
p.add_argument('--test_only', type=bool, default=False, help='test only')
p.add_argument('--restart', type=bool, default=False, help='do not reload from checkpoints')

# General training options
p.add_argument('--batch_size', type=int, default=100000)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=10,
               help='Epoch interval until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Step interval until loss is printed.')
p.add_argument('--model_type', type=str, default='relu',
               help='Options currently are "relu" (all relu activations), "ffm" (fourier feature mapping),'
                    '"rff" (random fourier features)')
args = p.parse_args()

# prepare data loader
video = Video(skvideo.datasets.bikes())
video_dataset = VideoDataset(video)
train_dataloader = DataLoader(video_dataset, pin_memory=True, num_workers=8, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(video_dataset, pin_memory=False, num_workers=8, batch_size=args.batch_size, shuffle=True)

if args.model_type == 'relu':
    model = make_ffm_network(4, 1024).cuda()
elif args.model_type == 'ffm':
    B = torch.normal(0., 10., size=(256, 2))
    B_t = torch.normal(0., 10., size=(256, 1))
    model = make_ffm_network(4, 1024, B, B_t).cuda()
elif args.model_type == 'rff':
    raise NotImplementedError
else:
    raise NotImplementedError
logdir = args.logdir

# load checkpoints
if os.path.exists(os.path.join(logdir, 'checkpoints')):
    ckpts = [os.path.join(logdir, 'checkpoints', f) for f in sorted(os.listdir(os.path.join(logdir, 'checkpoints'))) if 'pt' in f]
    if len(ckpts) > 0 and not args.restart:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        model.load_state_dict(torch.load(ckpt_path))

# training
if not args.test_only:
    train(model, train_dataloader, args.lr, epochs=args.num_epochs, 
        logdir=logdir, epochs_til_checkpoint=args.epochs_til_ckpt, 
        steps_til_summary=args.steps_til_summary, val_dataloader=val_dataloader)

# make full testing
print("Running full validation set...")
val_dataloader = DataLoader(video_dataset, pin_memory=False, num_workers=val_dataloader.num_workers, 
    batch_size=args.batch_size, shuffle=False)

model.eval()
with torch.no_grad():
    preds = []
    for (model_input, gt) in tqdm(val_dataloader):
        model_input, gt = model_input.cuda(), gt.cuda()
        model_out = model_pred(model, model_input)
        preds.append(model_out.cpu().numpy())
        
preds = np.vstack(preds).reshape(video.shape + (video.channels,))
preds = np.clip((preds / 2. + 0.5) * 255, 0, 255).astype(np.uint8)
writer = skvideo.io.FFmpegWriter(os.path.join(logdir, "test.mp4"))
for i in range(video.shape[0]):
        writer.writeFrame(preds[i, :, :, :])
writer.close()
