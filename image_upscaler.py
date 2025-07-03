import torch
from torchvision import transforms

from PIL import Image

from datetime import datetime

from models.RRDBNet_arch import RRDBNet

print("Where is your image located?")
file_directory = input()

with Image.open(file_directory) as im:
    transform = transforms.ToTensor()
    tensor = transform(im).unsqueeze(0)

    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
    model.load_state_dict(torch.load('models/RRDB_ESRGAN_x4.pth', map_location='cpu'))
    model.eval()

    with torch.no_grad():
        output = model(tensor)

        output_image = output.squeeze(0)                     # Remove batch dim → [3, H, W]
        output_image = output_image.clamp(0, 1)              # Ensure values are in [0,1]
        output_image = output_image.mul(255).byte()          # Scale to [0,255] and convert to int
        output_image = output_image.permute(1, 2, 0).cpu()   # Reorder to [H, W, C]

        output_pil = Image.fromarray(output_image.numpy())   # Convert to PIL image

        now = datetime.now()

        formatted = now.strftime("%d_%B_%Y_%H") + ".png"

        output_pil.save(formatted)
