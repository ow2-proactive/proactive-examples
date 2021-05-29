
print("BEGIN DesNet_161")

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
cnn = models.densenet161(pretrained=""" + str(pretrained) + """)
cnn.features = nn.Sequential(*list(cnn.children())[:-1])
cnn.classifier = nn.Linear(2208, num_classes)
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

print("END DesNet_161")