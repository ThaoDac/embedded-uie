import pytorch_lightning as pl
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import kornia.metrics.psnr as PSNR
import kornia.metrics.ssim as SSIM
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from loss.L1 import L1_loss
from loss.Perceptual_our import PerceptualLoss
from loss.UCR import UnContrastLoss, mosaic_module

from argparse import Namespace
from dataloader3 import Haze4kdataset, Val4kdataset
from pytorch_lightning import seed_everything
from network import Network
# Set seed
seed = 42  # Global seed set to 42
seed_everything(seed)
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger('tb_logs', name='UCR')

class CoolSystem(pl.LightningModule):

    def __init__(self, hparams):
        super(CoolSystem, self).__init__()

        self.params = hparams

        # train/val/test datasets
        self.train_datasets = self.params.train_datasets
        self.train_batchsize = self.params.train_bs
        self.test_datasets = self.params.test_datasets
        self.test_batchsize = self.params.test_bs
        self.validation_datasets = self.params.val_datasets
        self.val_batchsize = self.params.val_bs

        # Train setting
        self.initlr = self.params.initlr  # initial learning
        self.weight_decay = self.params.weight_decay  # optimizers weight decay
        self.crop_size = self.params.crop_size  # random crop size
        self.num_workers = self.params.num_workers

        # loss_function
        self.loss_L1 = L1_loss()
        self.loss_Pe = PerceptualLoss()
        self.UCR = UnContrastLoss()
        self.model = Network()

    def forward(self, x):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        y_lable = self.model(x1)
        return y_lable
    
    def forward1(self, x):
        y = self.model(x)
        return y

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.initlr, betas=[0.9, 0.999],
                                      weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.initlr, max_lr=1.5 * self.initlr,
                                                      cycle_momentum=False)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1 = torch.clip(x1, 0, 1)
        x2 = torch.clip(x2, 0, 1)
        x3 = mosaic_module(x2, 16, 16)
        y = torch.clip(y, 0, 1)
        y2 = self.forward(x)
        loss = self.loss_L1(y2,y) + 0.2 * self.loss_Pe(y2, y) + 0.2 * self.UCR(y2, y, x3)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward1(x)
        loss = self.loss_L1(y_hat, y) + 0.2 * self.loss_Pe(y_hat, y)
        ssim = SSIM(y_hat, y, 5).mean().item()
        psnr = PSNR(y_hat, y, 1).item()

        # Log metrics
        self.log('val_loss', loss)
        self.log('psnr', psnr)
        self.log('ssim', ssim)

        return {'val_loss': loss, 'psnr': psnr, 'ssim': ssim}

    def train_dataloader(self):
        # REQUIRED
        train_set = Haze4kdataset(self.train_datasets, train=True, size=self.crop_size)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.train_batchsize, shuffle=True,
                                                   num_workers=self.num_workers)
        return train_loader

    def val_dataloader(self):
        val_set = Val4kdataset(self.validation_datasets, train=False)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.val_batchsize, shuffle=False,
                                                 num_workers=self.num_workers)
        return val_loader


def main():
    RESUME = False
    resume_checkpoint_path = r''
    device = [int(x) for x in str(sys.argv[1]).split(',')]
    print(device)
    args = {
        'epochs': 100,
        # datasets - Updated to use PGHS dataset
        'train_datasets': r'/home/ndpthao/eject/IMPLEMENTATION/PGHS-main/Dataset/train',
        'test_datasets': None,
        'val_datasets': r'/home/ndpthao/eject/IMPLEMENTATION/PGHS-main/Dataset/train',  # Sử dụng train làm val nếu chưa có val folder riêng
        # bs
        'train_bs': 8,
        # 'train_bs':4,

        'test_bs': 1,
        'val_bs': 8,
        # 'initlr':0.0002,
        'initlr': 0.0001,
        'weight_decay': 0.001,
        'crop_size': 256,
        'num_workers': 16,
        # Net
        'model_blocks': 5,
        'chns': 64
    }

    hparams = Namespace(**args)

    model = CoolSystem(hparams)

    # Callback để lưu model best dựa trên PSNR (giá trị càng cao càng tốt)
    checkpoint_callback_psnr = ModelCheckpoint(
        dirpath='./checkpoints/best_psnr',  # Thư mục lưu model có PSNR tốt nhất
        monitor='psnr',
        filename='best-epoch{epoch:02d}-psnr{psnr:.3f}-ssim{ssim:.3f}',
        auto_insert_metric_name=False,
        save_top_k=5,  # Lưu 5 model có PSNR tốt nhất
        mode="max"
    )

    # Callback để lưu model best dựa trên validation loss (giá trị càng thấp càng tốt)
    checkpoint_callback_val_loss = ModelCheckpoint(
        dirpath='./checkpoints/best_val_loss',  # Thư mục lưu model có val_loss tốt nhất
        monitor='val_loss',
        filename='best-epoch{epoch:02d}-val_loss{val_loss:.3f}-psnr{psnr:.3f}',
        auto_insert_metric_name=False,
        save_top_k=5,  # Lưu 5 model có val_loss tốt nhất
        mode="min"
    )

    # Callback để lưu model theo từng epoch
    checkpoint_callback_every_epoch = ModelCheckpoint(
        dirpath='./checkpoints/every_epoch',  # Thư mục lưu model theo từng epoch
        filename='epoch{epoch:02d}-psnr{psnr:.3f}-ssim{ssim:.3f}-val_loss{val_loss:.3f}',
        auto_insert_metric_name=False,
        every_n_epochs=10,  # Lưu mỗi 10 epoch
        save_top_k=-1  # Lưu tất cả
    )

    # Callback để lưu model cuối cùng
    checkpoint_callback_last = ModelCheckpoint(
        dirpath='./checkpoints',
        filename='last-epoch{epoch:02d}',
        auto_insert_metric_name=False,
        save_last=True
    )

    # Khởi tạo Trainer
    trainer = pl.Trainer(
        max_epochs=hparams.epochs,
        accelerator='gpu',
        devices=device,
        logger=logger,
        strategy='ddp' if len(device) > 1 else 'auto',
        precision='16-mixed',
        callbacks=[
            checkpoint_callback_psnr,
            checkpoint_callback_val_loss,
            checkpoint_callback_every_epoch,
            checkpoint_callback_last
        ],
    )

    # Bắt đầu training (có thể resume từ checkpoint nếu cần)
    if RESUME:
        trainer.fit(model, ckpt_path=resume_checkpoint_path)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    main()
