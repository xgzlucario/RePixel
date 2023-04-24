import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torchvision.models import vgg16


# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.prelu(out)
        out = self.conv2(out)
        return out + identity


num_residual_blocks = 16

# 定义 ESRGAN 生成器网络
class Generator(nn.Module):
    def __init__(self, scale_factor):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU(),
            *[ResidualBlock(64, 64, kernel_size=3, padding=1) for _ in range(num_residual_blocks)],
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(16, 3, kernel_size=9, padding=4),
        )

    def forward(self, x):
        return self.layers(x)


# 定义 ESRGAN 判别器网络
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.layers(x)
    

image_size = 512

# 自定义数据集类
class ImageDataset(Dataset):
    def __init__(self, image_dir, scale_factor):
        self.image_dir = image_dir
        self.scale_factor = scale_factor
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

        self.lr_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size // scale_factor, image_size // scale_factor), interpolation=3),
            transforms.ToTensor(),
        ])

        self.hr_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        img = cv2.imread(self.image_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lr = self.lr_transform(img)
        hr = self.hr_transform(img)
        return lr, hr

    def __len__(self):
        return len(self.image_paths)
    

# 感知损失
def perceptual_loss(hr_imgs, fake_hr, vgg):
    hr_vgg_features = vgg(hr_imgs)
    fake_hr_vgg_features = vgg(fake_hr)
    return nn.MSELoss()(hr_vgg_features, fake_hr_vgg_features)


def main():
    # 参数设置
    epochs = 100
    batch_size = 2
    learning_rate = 0.0002
    scale_factor = 2
    image_dir = './output'
    device = torch.device('cuda')

    # 创建生成器和判别器
    generator = Generator(scale_factor).to(device)
    discriminator = Discriminator().to(device)

    # 定义损失函数和优化器
    adversarial_loss = nn.BCEWithLogitsLoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # 加载数据集
    dataset = ImageDataset(image_dir, scale_factor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # 加载预训练的VGG-16模型并提取特定层作为特征提取器
    vgg = vgg16().features[:18].to(device)
    for param in vgg.parameters():
        param.requires_grad = False

    for dir in ["gen", "models"]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # 训练过程
    for epoch in range(epochs):
        for i, (lr_imgs, hr_imgs) in enumerate(dataloader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            batch_size = lr_imgs.size(0)

            # 训练判别器
            d_optimizer.zero_grad()
            real_preds = discriminator(hr_imgs)
            fake_hr = generator(lr_imgs)
            fake_preds = discriminator(fake_hr.detach())

            real_labels = torch.ones_like(real_preds).to(device)
            fake_labels = torch.zeros_like(fake_preds).to(device)

            real_loss = adversarial_loss(real_preds, real_labels)
            fake_loss = adversarial_loss(fake_preds, fake_labels)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()

            # 训练生成器
            g_optimizer.zero_grad()
            fake_hr = generator(lr_imgs)
            fake_preds = discriminator(fake_hr)

            # 计算感知损失
            percep_loss = perceptual_loss(hr_imgs, fake_hr, vgg)
            adv_loss = adversarial_loss(fake_preds, real_labels)
            g_loss = percep_loss + 1e-3 * adv_loss
            g_loss.backward()
            g_optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}] | Batch [{i+1}/{len(dataloader)}] | D_loss: {d_loss.item()} | G_loss: {g_loss.item()}")

        # 保存生成的图像
        save_image(fake_hr[:25], f"gen/epoch_{epoch+1}.png", nrow=5, normalize=True)

        # 保存模型
        torch.save(generator.state_dict(), f"models/generator_epoch_{epoch+1}.pth")
        torch.save(discriminator.state_dict(), f"models/discriminator_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()

