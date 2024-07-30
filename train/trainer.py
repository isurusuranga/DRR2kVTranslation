import os
import torch
import torch.nn as nn
import torchvision.utils as utils
from models import Discriminator, Generator
from datasets import *
from utils import *
import itertools


class Trainer(object):
    def __init__(self, options):
        self.options = options
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.saver = CheckpointSaver(save_dir=options.model_save_path)

        trainDRR = os.path.join(self.options.dataroot, 'trainDRR')
        trainKV = os.path.join(self.options.dataroot, 'trainkV')
        testDRR = os.path.join(self.options.dataroot, 'testDRR')
        testKV = os.path.join(self.options.dataroot, 'testkV')

        self.train_dataset = ImageDataset(rootDRR=trainDRR, rootkV=trainKV, opt=self.options)
        self.test_dataset = ImageDataset(rootDRR=testDRR, rootkV=testKV, opt=self.options)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.options.batch_size,
                                                        shuffle=True, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.options.batch_size,
                                                       shuffle=True, pin_memory=True)

        self.netG_A2B = Generator(options).to(self.device)
        self.netG_B2A = Generator(options).to(self.device)
        self.netD_A = Discriminator(options).to(self.device)
        self.netD_B = Discriminator(options).to(self.device)

        self.cycle_loss = torch.nn.L1Loss().to(self.device)
        self.identity_loss = torch.nn.L1Loss().to(self.device)

        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
                                            lr=self.options.lr, betas=(0.5, 0.999))
        self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=self.options.lr, betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=self.options.lr, betas=(0.5, 0.999))

        lr_lambda = DecayLR(self.options.num_epochs, 0, self.options.decay_epochs).step

        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lr_lambda)
        self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_A, lr_lambda=lr_lambda)
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_B, lr_lambda=lr_lambda)

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        self.checkpoint = None

        if self.options.resume and self.saver.exists_checkpoint():
            self.checkpoint = self.saver.load_checkpoint()
            print("=> loaded checkpoint '{}' (epoch {})".format(self.saver.get_checkpoint(),
                                                                self.checkpoint['last_epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(self.saver.get_checkpoint()))

    def load_models_state(self):
        self.netG_A2B.load_state_dict(self.checkpoint['netG_A2B_state_dict'])
        self.netG_B2A.load_state_dict(self.checkpoint['netG_B2A_state_dict'])
        self.netD_A.load_state_dict(self.checkpoint['netD_A_state_dict'])
        self.netD_B.load_state_dict(self.checkpoint['netD_B_state_dict'])

    def load_optimizers_state(self):
        self.optimizer_G.load_state_dict(self.checkpoint['optimizer_G_state_dict'])
        self.optimizer_D_A.load_state_dict(self.checkpoint['optimizer_D_A_state_dict'])
        self.optimizer_D_B.load_state_dict(self.checkpoint['optimizer_D_B_state_dict'])

    def load_schedulers_state(self):
        self.lr_scheduler_G.load_state_dict(self.checkpoint['lr_scheduler_G_state_dict'])
        self.lr_scheduler_D_A.load_state_dict(self.checkpoint['lr_scheduler_D_A_state_dict'])
        self.lr_scheduler_D_B.load_state_dict(self.checkpoint['lr_scheduler_D_B_state_dict'])

    def sample_images_test_batch(self, epoch_no):
        """Saves a generated sample from the test set"""
        imgs = next(iter(self.test_loader))

        self.netG_A2B.eval()
        self.netG_B2A.eval()

        with torch.no_grad():
            real_image_A = imgs["A"]["img"].to(self.device)
            real_image_A_gantry = imgs["A"]["gantry"].to(self.device)
            fake_image_B = self.netG_A2B(real_image_A, real_image_A_gantry)

            real_image_B = imgs["B"]["img"].to(self.device)
            real_image_B_gantry = imgs["B"]["gantry"].to(self.device)
            fake_image_A = self.netG_B2A(real_image_B, real_image_B_gantry)
            recovered_image_A = self.netG_B2A(fake_image_B, real_image_A_gantry)
            recovered_image_B = self.netG_A2B(fake_image_A, real_image_B_gantry)

        # Arrange images along x-axis
        real_A = utils.make_grid(real_image_A, nrow=5, normalize=True)
        real_B = utils.make_grid(real_image_B, nrow=5, normalize=True)
        fake_A = utils.make_grid(fake_image_A, nrow=5, normalize=True)
        fake_B = utils.make_grid(fake_image_B, nrow=5, normalize=True)
        recon_A = utils.make_grid(recovered_image_A, nrow=5, normalize=True)
        recon_B = utils.make_grid(recovered_image_B, nrow=5, normalize=True)

        # Arrange images along y-axis
        image_grid = torch.cat((real_A, fake_B, recon_A, real_B, fake_A, recon_B), 1)
        utils.save_image(image_grid, "%s/%s.png" % (self.options.output_dir, epoch_no), normalize=False)

    def model_train(self):
        start_epoch = 0
        print_freq = 5

        if self.checkpoint is not None:
            start_epoch = self.checkpoint['last_epoch'] + 1
            self.load_models_state()
            self.load_optimizers_state()
            self.load_schedulers_state()

        for epoch in range(start_epoch, self.options.num_epochs):

            for i, data in enumerate(self.train_loader):
                real_image_A = data["A"]["img"].to(self.device)
                real_image_A_gantry = data["A"]["gantry"].to(self.device)

                real_image_B = data["B"]["img"].to(self.device)
                real_image_B_gantry = data["B"]["gantry"].to(self.device)

                ##############################################
                # (1) Update G network: Generators A2B and B2A
                ##############################################
                self.netG_B2A.train()
                self.netG_A2B.train()

                # Set G_A and G_B's gradients to zero
                self.optimizer_G.zero_grad()

                # Identity loss
                # G_B2A(A) should equal A if real A is fed
                identity_image_A = self.netG_B2A(real_image_A, real_image_A_gantry)
                loss_identity_A = self.identity_loss(identity_image_A, real_image_A) * self.options.ilw
                # G_A2B(B) should equal B if real B is fed
                identity_image_B = self.netG_A2B(real_image_B, real_image_B_gantry)
                loss_identity_B = self.identity_loss(identity_image_B, real_image_B) * self.options.ilw

                # GAN loss
                # GAN loss D_A(G_A(A))
                fake_image_A = self.netG_B2A(real_image_B, real_image_B_gantry)
                fake_output_A = self.netD_A(fake_image_A, real_image_B_gantry)
                loss_GAN_B2A = -fake_output_A.mean()
                # GAN loss D_B(G_B(B))
                fake_image_B = self.netG_A2B(real_image_A, real_image_A_gantry)
                fake_output_B = self.netD_B(fake_image_B, real_image_A_gantry)
                loss_GAN_A2B = -fake_output_B.mean()

                # Cycle loss
                recovered_image_A = self.netG_B2A(fake_image_B, real_image_A_gantry)
                loss_cycle_ABA = self.cycle_loss(recovered_image_A, real_image_A) * self.options.clw

                recovered_image_B = self.netG_A2B(fake_image_A, real_image_B_gantry)
                loss_cycle_BAB = self.cycle_loss(recovered_image_B, real_image_B) * self.options.clw

                # Combined loss and calculate gradients
                errG = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

                # Calculate gradients for G_A and G_B
                errG.backward()
                # Update G_A and G_B's weights
                self.optimizer_G.step()

                ##############################################
                # (2) Update D network: Discriminator A
                ##############################################

                # Set D_A gradients to zero
                self.optimizer_D_A.zero_grad()

                # Real A image loss
                real_output_A = self.netD_A(real_image_A, real_image_A_gantry)
                errD_real_A = nn.ReLU()(1.0 - real_output_A).mean()

                # Fake A image loss
                # push this to the buffer {"img":fake_image_A, "gantry":real_image_B_gantry}
                fake_image_A = self.fake_A_buffer.push_and_pop(fake_image_A)
                fake_output_A = self.netD_A(fake_image_A.detach(), real_image_B_gantry)
                errD_fake_A = nn.ReLU()(1.0 + fake_output_A).mean()

                # Combined loss and calculate gradients
                errD_A = (errD_real_A + errD_fake_A) / 2

                # Calculate gradients for D_A
                errD_A.backward()
                # Update D_A weights
                self.optimizer_D_A.step()

                ##############################################
                # (3) Update D network: Discriminator B
                ##############################################

                # Set D_B gradients to zero
                self.optimizer_D_B.zero_grad()

                # Real B image loss
                real_output_B = self.netD_B(real_image_B, real_image_B_gantry)
                errD_real_B = nn.ReLU()(1.0 - real_output_B).mean()

                # Fake B image loss
                fake_image_B = self.fake_B_buffer.push_and_pop(fake_image_B)
                fake_output_B = self.netD_B(fake_image_B.detach(), real_image_A_gantry)
                errD_fake_B = nn.ReLU()(1.0 + fake_output_B).mean()

                # Combined loss and calculate gradients
                errD_B = (errD_real_B + errD_fake_B) / 2

                # Calculate gradients for D_B
                errD_B.backward()
                # Update D_B weights
                self.optimizer_D_B.step()

                batches_done = epoch * len(self.train_loader) + i

                # print at the end of each epoch
                if batches_done != 0 and (batches_done + 1) % len(self.train_loader) == 0:
                    print(
                        f"[{epoch}/{self.options.num_epochs - 1}][{i}/{len(self.train_loader) - 1}] "
                        f"Loss_D: {(errD_A + errD_B).item():.4f} "
                        f"Loss_G: {errG.item():.4f} "
                        f"Loss_G_identity: {(loss_identity_A + loss_identity_B).item():.4f} "
                        f"loss_G_GAN: {(loss_GAN_A2B + loss_GAN_B2A).item():.4f} "
                        f"loss_G_cycle: {(loss_cycle_ABA + loss_cycle_BAB).item():.4f}")

            self.sample_images_test_batch(epoch)
            state = {
                'last_epoch': epoch,
                'netG_A2B_state_dict': self.netG_A2B.state_dict(),
                'netG_B2A_state_dict': self.netG_B2A.state_dict(),
                'netD_A_state_dict': self.netD_A.state_dict(),
                'netD_B_state_dict': self.netD_B.state_dict(),
                'optimizer_G_state_dict': self.optimizer_G.state_dict(),
                'optimizer_D_A_state_dict': self.optimizer_D_A.state_dict(),
                'optimizer_D_B_state_dict': self.optimizer_D_B.state_dict(),
                'lr_scheduler_G_state_dict': self.lr_scheduler_G.state_dict(),
                'lr_scheduler_D_A_state_dict': self.lr_scheduler_D_A.state_dict(),
                'lr_scheduler_D_B_state_dict': self.lr_scheduler_D_B.state_dict()
            }
            self.saver.save_checkpoint(state)

            if (epoch % print_freq == 0) and epoch >= 40:
                self.saver.save_temp_checkpoint(state)

            # Update learning rates
            self.lr_scheduler_G.step()
            self.lr_scheduler_D_A.step()
            self.lr_scheduler_D_B.step()
