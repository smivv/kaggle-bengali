import os
import numpy as np
import torch
import safitty

from PIL import Image
from torch.nn.functional import softmax
from torchvision.transforms import Compose, Resize, Normalize, ToTensor

from catalyst.utils import load_config

from src.models.cico.generic import GenericModel

logs = "/home/smirnvla/PycharmProjects/catalyst-classification/logs/cico15_256_arcface_64_5e-1_radam_plateu_1e-0_l2neck_stratified/"

yml_path = logs + "configs/cico.yml"
pth_path = logs + "checkpoints/best.pth"
pb_path = logs + "model.pb"
jit_path = logs + "traced.jit"
img_path = "/workspace/Datasets/CICO1.5/benchmarking_plan/v2/test/Pizza__/1_3_11_Camera_7_121.jpg"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


config = load_config(yml_path)

model = GenericModel.get_from_params(
    backbone_params=safitty.get(config, 'model_params', 'backbone_params'),
    neck_params=safitty.get(config, 'model_params', 'neck_params'),
    heads_params=safitty.get(config, 'model_params', 'heads_params')
)

checkpoint = torch.load(pth_path, map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# torch.save(model, pb_path)
torch.jit.save(torch.jit.trace(model, torch.rand(1, 3, 224, 224)), jit_path)
model = torch.jit.load(jit_path)
# model = torch.load(pb_path)
# model.to(device)

mean, std = 0.449, 0.226

transforms = Compose([
    Resize(224),
    ToTensor(),
    Normalize(mean=0.449, std=0.226),
])

image = Image.open(img_path).convert("RGB")
image = transforms(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)
    output = softmax(output)
    print(output)
