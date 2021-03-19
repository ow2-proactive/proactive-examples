print("BEGIN SegNet")

import json
from ast import literal_eval as make_tuple

IMG_SIZE = variables.get("IMG_SIZE")
NUM_CLASSES = int(str(variables.get("NUM_CLASSES")))
 
IMG_SIZE = make_tuple(IMG_SIZE)

if NUM_CLASSES == 2:
    NUM_CLASSES = NUM_CLASSES - 1       
else:
    NUM_CLASSES = NUM_CLASSES + 2   

# Define the NET model
NET_MODEL = """

class SegNetEnc(nn.Module):

    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()

        layers = [
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ] * num_layers
        layers += [
            nn.Conv2d(in_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class SegNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        # should be vgg16bn but at the moment we have no pretrained bn models
        decoders = list(models.vgg16(pretrained=True).features.children())

        self.dec1 = nn.Sequential(*decoders[:5])
        self.dec2 = nn.Sequential(*decoders[5:10])
        self.dec3 = nn.Sequential(*decoders[10:17])
        self.dec4 = nn.Sequential(*decoders[17:24])
        self.dec5 = nn.Sequential(*decoders[24:])

        # gives better results
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.requires_grad = False

        self.enc5 = SegNetEnc(512, 512, 1)
        self.enc4 = SegNetEnc(1024, 256, 1)
        self.enc3 = SegNetEnc(512, 128, 1)
        self.enc2 = SegNetEnc(256, 64, 0)
        self.enc1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, 3, padding=1)

    def forward(self, x):
        dec1 = self.dec1(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)
        dec5 = self.dec5(dec4)
        enc5 = self.enc5(dec5)
        enc4 = self.enc4(torch.cat([dec4, enc5], 1))
        enc3 = self.enc3(torch.cat([dec3, enc4], 1))
        enc2 = self.enc2(torch.cat([dec2, enc3], 1))
        enc1 = self.enc1(torch.cat([dec1, enc2], 1))

        return F.upsample_bilinear(self.final(enc1), x.size()[2:])
        
        
METHOD_NAME = 'SegNet'        
Net = SegNet    
assert Net is not None, f'model {MODEL_NAME} not available'
model = Net(NUM_CLASSES)
        
"""
print(NET_MODEL)

# Data augmentation and normalization for training
# Just normalization for validation and test
NET_TRANSFORM = """
import numpy as np

def colormap(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))

        cmap[i,:] = np.array([r, g, b])

    return cmap

class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert isinstance(tensor, torch.LongTensor), 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor


class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)


class Colorize:

    def __init__(self, n=22):
        self.cmap = colormap(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(1, len(self.cmap)):
            mask = gray_image[0] == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
       
if NUM_CLASSES == 1:
    input_transform = Compose([Resize(IMG_SIZE),ToTensor()])
    target_transform = Compose([Resize(IMG_SIZE),ToTensor()])
else:
    color_transform = Colorize()
    image_transform = ToPILImage()
    input_transform = Compose([
            Resize(IMG_SIZE),   
            ToTensor(),
            Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    target_transform = Compose([
            Resize(IMG_SIZE),     
            ToLabel(),
            Relabel(255, 21),
    ])
    
"""
print(NET_TRANSFORM)

#CRITERION FUNCTION
NET_CRITERION = """
class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = nn.NLLLoss(weight)
        
    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs), targets)

class BinaryCrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super().__init__()
        
        #self.loss = nn.BCELoss(weight)
        self.loss = nn.BCEWithLogitsLoss(weight)

    def forward(self, outputs, targets):
        #return self.loss(nn.Sigmoid(outputs), targets)
        return self.loss(outputs, targets)
        
"""
print(NET_CRITERION)


if 'variables' in locals():
  variables.put("NET_MODEL", NET_MODEL)
  variables.put("NET_TRANSFORM", NET_TRANSFORM)
  variables.put("NET_CRITERION", NET_CRITERION)
  variables.put("IMG_SIZE", IMG_SIZE)
  variables.put("NUM_CLASSES", NUM_CLASSES)

print("END SegNet")