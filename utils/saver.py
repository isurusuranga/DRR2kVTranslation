import os
import torch


class CheckpointSaver(object):
    """Class that handles saving and loading checkpoints during training."""

    def __init__(self, save_dir):
        self.save_dir = os.path.abspath(save_dir)
        self.temp_dir = os.path.join(self.save_dir, 'temp')

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        self.checkpoint = self.get_checkpoint()

    def exists_checkpoint(self):
        """Check if a checkpoint exists in the current directory."""
        status = False if self.checkpoint is None else True
        return status

    def save_checkpoint(self, state):
        filename = '{}/DRR2kV_Cycle_GAN.pth'.format(self.save_dir)

        torch.save(state, filename)

    def save_temp_checkpoint(self, state):
        custom_save_path = 'DRR2kV_Cycle_GAN_' + str(state['last_epoch']) + '.pth'
        filename = os.path.join(self.temp_dir, custom_save_path)

        torch.save(state, filename)

    def load_checkpoint(self):
        """Load a checkpoint."""
        checkpoint = torch.load(self.checkpoint)

        return checkpoint

    def get_checkpoint(self):
        """Get filename of latest checkpoint if it exists."""
        checkpoint_save_path = os.path.abspath(os.path.join(self.save_dir, 'DRR2kV_Cycle_GAN.pth'))
        checkpoint = checkpoint_save_path if (os.path.exists(checkpoint_save_path)) else None

        return checkpoint





