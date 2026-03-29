import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class UnpairedDataset(Dataset):
    def __init__(self, echo_dir, mri_dir, size=256):
        self.echo_files = sorted([
            os.path.join(echo_dir, f)
            for f in os.listdir(echo_dir)
            if f.endswith(('.png', '.jpg'))
        ])
        self.mri_files = sorted([
            os.path.join(mri_dir, f)
            for f in os.listdir(mri_dir)
            if f.endswith(('.png', '.jpg'))
        ])
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        print(f"Echo images: {len(self.echo_files)}")
        print(f"MRI  images: {len(self.mri_files)}")

    def __len__(self):
        return max(len(self.echo_files), len(self.mri_files))

    def __getitem__(self, idx):
        echo = Image.open(self.echo_files[idx % len(self.echo_files)])
        mri  = Image.open(self.mri_files[idx  % len(self.mri_files)])
        return self.transform(echo), self.transform(mri)