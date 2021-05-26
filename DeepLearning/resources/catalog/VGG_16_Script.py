# Copyright Activeeon 2007-2021. All rights reserved.
print("BEGIN VGG_16")

import json

USE_PRETRAINED_MODEL = 'true'
if 'variables' in locals():
  if variables.get("USE_PRETRAINED_MODEL") is not None:
    USE_PRETRAINED_MODEL = str(variables.get("USE_PRETRAINED_MODEL")).lower()

pretrained = False
if USE_PRETRAINED_MODEL == 'true':
  pretrained = True

# Define the CNN model
CNN_MODEL = """
cnn = models.vgg16(pretrained=""" + str(pretrained) + """)
num_ftrs = cnn.classifier[6].in_features
feature_model = list(cnn.classifier.children())
feature_model.pop()
feature_model.append(nn.Linear(num_ftrs, num_classes))
cnn.classifier = nn.Sequential(*feature_model) 
"""
print(CNN_MODEL)

# Data augmentation and normalization for training
# Just normalization for validation and test
CNN_TRANSFORM = """
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
"""
print(CNN_TRANSFORM)

if 'variables' in locals():
  variables.put("CNN_MODEL", CNN_MODEL)
  variables.put("CNN_TRANSFORM", CNN_TRANSFORM)

print("END VGG_16")