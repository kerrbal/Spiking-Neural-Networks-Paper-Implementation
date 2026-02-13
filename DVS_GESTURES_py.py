# %%
#Here DVS_Gestures dataset is trained, I used (64x64) pixels, because it is not so small and enough for my gpu
import tonic
import tonic.transforms as transforms
from PROJECT.Trace import get_model_eligibility, train_eligibility
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


ds = tonic.transforms.Downsample(
    sensor_size=tonic.datasets.DVSGesture.sensor_size,
    target_size=(64, 64)  # 128->64
)

transform = transforms.Compose([
    ds,
    transforms.ToFrame(
        sensor_size=(64, 64, 2),
        n_time_bins=20 #only 20 time steps
    ),
])
#download dataset
train_dataset = tonic.datasets.DVSGesture(
    save_to='./Data3',
    train=True,
    transform=transform
)

test_dataset = tonic.datasets.DVSGesture(
    save_to='./Data3',
    train = False,
    transform=transform
)


events, label = train_dataset[0]
print(f"Events shape: {events.shape}")
print(f"Events dtype: {events.dtype}")
# shape -> [x, y, t, polarity]
# My functions take (T, B, in_dim) as input, so
# I can flatten the dataset to arrange the shapes.


# %%
the_device = torch.device("cuda")
tonic.datasets.DVSGesture.sensor_size

# %%
def collate_fn(batch):
    """
    Since Time steps are different, I used padding, however when I changed transform  to n_time_bins value, so
    it is no padding, in the general sense. It just holds the values, it doesnt padd anything since therei s only fixed t values
    """
    frames_list, labels = zip(*batch)
    
    #maxt t
    max_T = max(f.shape[0] for f in frames_list)
    B = len(frames_list)
    
    
    polarity, H, W = frames_list[0].shape[1:]
    in_features = polarity * H * W  # 2 * 32 * 32 = 2048, I flattened them because my functions take (T, B, h*w) as input
    
    padded = torch.zeros(max_T, B, in_features)
    
    for i, f in enumerate(frames_list):
        T = f.shape[0]
        flat = torch.tensor(f).float().reshape(T, -1)  # (T, C*H*W)
        padded[:T, i, :] = flat
    
    labels = torch.tensor(labels)
    return padded, labels

# %%
batch_size = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,  # Windows'ta 0 olmalı
    drop_last=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0
)

# Test et
for frames, labels in train_loader:
    print(f"Batch frames shape: {frames.shape}")  # (T, B, 2048)
    print(f"Batch labels shape: {labels.shape}")  # (B,)
    break

# %%
from PROJECT.Trace import get_model_eligibility, train_eligibility

epoch = 2
model, optimizer = get_model_eligibility([8192, 2048, 1024, 11], thresholds = [5, 4, 3], lr = 0.005, out_dims=11)
for i in model.G_hiddens:
    print(i.shape)
model = model.to(the_device)
model, acc, spikes_means = train_eligibility(train_loader, test_loader, model, optimizer, num_epochs=epoch, verbose=True, apply_poisson=True)
print(f"accuracy is --> ", acc)
spikes_means = torch.mean(spikes_means, dim = 0, keepdim=False)
for i in range(len(spikes_means)):
    print(f"Layer {i} density -> ", spikes_means[i])
print()


# %%
for i in model.lif_layers:
    print(i.W.device)


