from PIL import Image
import numpy as np


def save_image(predicted_img, save_path):
    out = (predicted_img[0] * 0.5) + 0.5
    out = out.cpu().clone().detach().numpy()
    out = np.transpose(out, (1, 2, 0))
    out = np.squeeze(out, axis=2)

    out = out * 255.
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)
    out = Image.fromarray(out)

    out.save(save_path)
