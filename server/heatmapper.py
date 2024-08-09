
import cv2
import torch
from torch import nn
from typing import Tuple
from torchvision import transforms as T
from torchvision.models.mobilenetv3 import mobilenet_v3_large

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.preprocess = T.Compose([T.Resize(768),T.ToTensor(),T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.postprocess = T.Compose([T.Resize(768),T.ToTensor()])
        self.layer_name = "features.16.0"
        self.model = mobilenet_v3_large(True).eval().cpu()
        self.hooked = {}
        
    def forward(self,x):
        hook = self.model.features[16][0].register_forward_hook(self._forward_hook)
        tensor = self.preprocess(x).unsqueeze(0).cpu()
        output = self.model(tensor)
        feature = self.hooked['output']
        h,w = output.shape
        hook.remove()
        _, _, vT = torch.linalg.svd(feature)
        v1 = vT[:, :, 0, :][..., None, :]
        heatmap = feature @ v1.repeat(1, 1, v1.shape[3], 1)
        heatmap = heatmap.sum(1)
        heatmap -= heatmap.min()
        heatmap = heatmap / heatmap.max() * 255
        original = self.postprocess(x) * 255
        repeated = T.Resize(size=(original.shape[1],original.shape[2]),interpolation=T.InterpolationMode.BICUBIC)(torch.repeat_interleave(heatmap,3,0))
        
        original = original.permute(1,2,0).detach().cpu().numpy().astype('uint8')
        repeated = repeated.permute(1,2,0).detach().cpu().numpy().astype('uint8')
        colorCoded = cv2.applyColorMap(repeated, cv2.COLORMAP_JET) 
        composite = original * 0.7 + colorCoded * 0.3

        return composite
    
    def _forward_hook(self, module, inputs: Tuple[torch.Tensor], outputs):
        self.hooked['output'] = outputs
