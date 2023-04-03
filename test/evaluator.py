import torch
import os
from PIL import Image
from models import Generator
import torchvision.transforms.functional as TF
from utils import *
import glob


class Evaluator(object):
    def __init__(self, options):
        self.options = options
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir)

        # pass the absolute folder path which exists all the images we need to transfer the style
        # Point to each DRR folder (i.e. train/validation/test) that need to transferred style to generate synthetic kVs
        self.test_drr_paths = glob.glob(os.path.join(self.options.dataroot, '*.png'))

        # create drr2kV generator model
        self.netG_A2B = Generator(options).to(self.device)
        self.checkpoint = self.saver.load_checkpoint()
        # Load state dicts
        self.netG_A2B.load_state_dict(self.checkpoint['netG_A2B_state_dict'])

    def evaluate(self):
        # Set model evaluate mode
        self.netG_A2B.eval()

        # generate fake kV from real DRR images
        for i, path in enumerate(self.test_drr_paths):
            # extract the drr image name with extension from the absolute path (e.g. volume-10_0.png)
            img_name_str = os.path.basename(path).rsplit('.', 1)[0]
            drr_gantry_angle = float(os.path.basename(img_name_str).rsplit('_', 1)[1])
            drr_gantry_angle = torch.tensor([drr_gantry_angle / 360]).to(self.device)

            # Path should be come with rowdata for a given deformed graph
            img = Image.open(path)
            img = img.resize((self.options.img_res, self.options.img_res))
            img = TF.to_tensor(img)
            img = TF.normalize(img, (0.5,), (0.5,)).to(self.device)
            # need to provide the batch dimension at dim0 (e.g. torch.Size([1, 1, 256, 256]))
            img.unsqueeze_(0)

            with torch.no_grad():
                predicted_kV = self.netG_A2B(img, drr_gantry_angle)
                # recovered_drr = netG_B2A(predicted_kV, drr_gantry_angle)

            # Save predicted kV image
            predicted_img_name = img_name_str
            predicted_img_path = os.path.join(self.options.test_results_dir, predicted_img_name + ".png")
            save_image(predicted_kV, predicted_img_path)
