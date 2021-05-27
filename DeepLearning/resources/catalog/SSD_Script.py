
print("BEGIN SSD")

import json
import numpy as np
from ast import literal_eval as make_tuple

IMG_SIZE = variables.get("IMG_SIZE") 
START_ITERATION = variables.get("START_ITERATION") 
MAX_ITERATION = int(str(variables.get("MAX_ITERATION"))) 
LR_STEPS = variables.get("LR_STEPS") 
LR_FACTOR = float(str(variables.get("LR_FACTOR")))  
GAMMA = variables.get("GAMMA") 
USE_PRETRAINED_MODEL = variables.get("USE_PRETRAINED_MODEL") 
NUM_CLASSES = int(str(variables.get("NUM_CLASSES")))  
MIN_SIZES = variables.get("MIN_SIZES")
MAX_SIZES = variables.get("MAX_SIZES")
LEARNING_RATE = float(str(variables.get("LEARNING_RATE")))  
MOMENTUM = float(str(variables.get("MOMENTUM")))   
WEIGHT_DECAY = float(str(variables.get("WEIGHT_DECAY"))) 
LABEL_PATH  = variables.get("LABEL_PATH")
NET_NAME = 'SSD'

IMG_SIZE = make_tuple(IMG_SIZE)
IMG_SIZE = tuple(IMG_SIZE)

LR_STEPS = make_tuple(LR_STEPS)
LR_STEPS = tuple(LR_STEPS)

# Define the NET model
NET_MODEL = """

from __future__ import division

# SSD300 CONFIGS
dataset = {
    'num_classes': NUM_CLASSES,
    'lr_steps': LR_STEPS,
    'max_iter': MAX_ITERATION, 
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': IMG_SIZE[0],
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': MIN_SIZES, 
    'max_sizes': MAX_SIZES,
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
}

VARIANCE = [0.1, 0.2]

class Detect(Function):
# ======================================================================================
#    At test time, Detect is the final layer of SSD.  Decode location preds,
#    apply non-maximum suppression to location predictions based on conf
#    scores and threshold to a top_k number of output predictions for both
#    confidence score and locations.
# ======================================================================================
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = VARIANCE

    def forward(self, loc_data, conf_data, prior_data):
# ======================================================================================
#        Args:
#            loc_data: (tensor) Loc preds from loc layers
#                Shape: [batch,num_priors*4]
#            conf_data: (tensor) Shape: Conf preds from conf layers
#                Shape: [batch*num_priors,num_classes]
#            prior_data: (tensor) Prior boxes and variances from priorbox layers
#                Shape: [1,num_priors,4]
# ======================================================================================
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            #print("loc_data[i].is_cuda",loc_data[i].is_cuda) # loc_data[i].is_cuda True
            #print("prior_data.is_cuda",prior_data.is_cuda) # prior_data.is_cuda False
            if loc_data[i].is_cuda == True and prior_data.is_cuda == False:
                prior_data = prior_data.cuda()
            
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                #if scores.dim() == 0:
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output


class PriorBox(object):

# =============================================================================
#     Compute priorbox coordinates in center-offset form for each source
#     feature map.  
# =============================================================================

    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.cfg = (dataset)   
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

class SSD(nn.Module):
# =============================================================================
#    Single Shot Multibox Architecture
#    The network is composed of a base VGG network followed by the
#    added multibox conv layers.  Each multibox layer branches into
#        1) conv2d for class conf scores
#        2) conv2d for localization predictions
#        3) associated priorbox layer to produce default bounding
#           boxes specific to the layer's feature map size.
#    See: https://arxiv.org/pdf/1512.02325.pdf for more details.
#
#    Args:
#        phase: (string) Can be "test" or "train"
#        size: input image size
#        base: VGG16 layers for input, size of either 300 or 500
#        extras: extra layers that feed to multibox loc and conf layers
#        head: "multibox head" consists of loc and conf conv layers
# =============================================================================

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        #self.cfg = (coco, voc)[num_classes == 21]
        self.cfg = (dataset)      
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward())
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
# =============================================================================
#       Applies network layers and ops on input image(s) x.
#
#        Args:
#            x: input image or batch of images. Shape: [batch,3,300,300].
#
#        Return:
#            Depending on phase:
#            test:
#                Variable(tensor) of output class label predictions,
#                confidence score, and corresponding location predictions for
#                each object detected. Shape: [batch,topk,7]
#
#            train:
#                list of concat outputs from:
#                    1: confidence layers, Shape: [batch*num_priors,num_classes]
#                    2: localization layers, Shape: [batch,num_priors*4]
#                    3: priorbox layers, Shape: [2,num_priors*4]     
# =============================================================================
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)
    
"""
print(NET_MODEL)

# Define CRITERION functions
NET_CRITERION = """

def point_form(boxes):
# =============================================================================
#    Convert prior_boxes to (xmin, ymin, xmax, ymax)
#    representation for comparison to point form ground truth data.
#    Args:
#        boxes: (tensor) center-size default boxes from priorbox layers.
#    Return:
#        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
# =============================================================================
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
# =============================================================================
#    Convert prior_boxes to (cx, cy, w, h)
#    representation for comparison to center-size form ground truth data.
#    Args:
#        boxes: (tensor) point_form boxes
#    Return:
#        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes. 
# =============================================================================
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h

def intersect2(box_a, box_b):
# =============================================================================  
#     We resize both tensors to [A,B,2] without new malloc:
#    [A,2] -> [A,1,2] -> [A,B,2]
#    [B,2] -> [1,B,2] -> [A,B,2]
#    Then we compute the area of intersect between box_a and box_b.
#    Args:
#      box_a: (tensor) bounding boxes, Shape: [A,4].
#      box_b: (tensor) bounding boxes, Shape: [B,4].
#    Return:
#      (tensor) intersection area, Shape: [A,B].
# =============================================================================

    A = box_a.size(0)
    B = box_b.size(0)
    #A = box_a.size
    #B = box_b.size   
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
# =============================================================================
#    Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
#    is simply the intersection over union of two boxes.  Here we operate on
#    ground truth boxes and default boxes.
#    E.g.:
#        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
#    Args:
#        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
#        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
#    Return:
#        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
# =============================================================================

    inter = intersect2(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
# =============================================================================
#    Match each prior box with the ground truth box of the highest jaccard
#    overlap, encode the bounding boxes, then return the matched indices
#    corresponding to both confidence and location preds.
#    Args:
#        threshold: (float) The overlap threshold used when mathing boxes.
#        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
#        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
#        variances: (tensor) Variances corresponding to each prior coord,
#            Shape: [num_priors, 4].
#        labels: (tensor) All the class labels for the image, Shape: [num_obj].
#        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
#        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
#        idx: (int) current batch index
#    Return:
#        The matched indices corresponding to 1)location and 2)confidence preds.
# =============================================================================
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior
    
def encode(matched, priors, variances):
# =============================================================================
#    Encode the variances from the priorbox layers into the ground truth boxes
#    we have matched (based on jaccard overlap) with the prior boxes.
#    Args:
#        matched: (tensor) Coords of ground truth for each prior in point-form
#            Shape: [num_priors, 4].
#        priors: (tensor) Prior boxes in center-offset form
#            Shape: [num_priors,4].
#        variances: (list[float]) Variances of priorboxes
#    Return:
#        encoded boxes (tensor), Shape: [num_priors, 4]
# =============================================================================

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    #g_wh = torch.log(g_wh) / variances[1]
    g_wh = torch.log(g_wh + 1e-10) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
# =============================================================================
#    Decode locations from predictions using priors to undo
#    the encoding we did for offset regression at train time.
#    Args:
#        loc (tensor): location predictions for loc layers,
#            Shape: [num_priors,4]
#        priors (tensor): Prior boxes in center-offset form.
#            Shape: [num_priors,4].
#        variances: (list[float]) Variances of priorboxes
#    Return:
#        decoded bounding box predictions
# =============================================================================

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
# =============================================================================
#    Utility function for computing log_sum_exp while determining
#    This will be used to determine unaveraged confidence loss across
#    all examples in a batch.
#    Args:
#        x (Variable(tensor)): conf_preds from conf layers
# =============================================================================
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
# =============================================================================
#    Apply non-maximum suppression at test time to avoid detecting too many
#    overlapping bounding boxes for a given object.
#    Args:
#        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
#        scores: (tensor) The class predscores for the img, Shape:[num_priors].
#        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
#        top_k: (int) The Maximum number of box preds to consider.
#    Return:
#        The indices of the kept boxes with respect to num_priors.
# =============================================================================
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


class MultiBoxLoss(nn.Module):
# =========================================================================================
#    SSD Weighted Loss Function
#    Compute Targets:
#        1) Produce Confidence Target Indices by matching  ground truth boxes
#           with (default) 'priorboxes' that have jaccard index > threshold parameter
#           (default threshold: 0.5).
#        2) Produce localization target by 'encoding' variance into offsets of ground
#           truth boxes and their matched  'priorboxes'.
#        3) Hard negative mining to filter the excessive number of negative examples
#           that comes with using a large number of default bounding boxes.
#           (default negative:positive ratio 3:1)
#    Objective Loss:
#        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
#        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
#        weighted by α which is set to 1 by cross val.
#        Args:
#            c: class confidences,
#            l: predicted boxes,
#            g: ground truth boxes
#            N: number of matched default boxes
#        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
# =========================================================================================

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        VARIANCE = [0.1, 0.2] 
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = VARIANCE  
    def forward(self, predictions, targets):
# =============================================================================
#        Multibox Loss
#        Args:
#            predictions (tuple): A tuple containing loc preds, conf preds,
#            and prior boxes from SSD net.
#                conf shape: torch.size(batch_size,num_priors,num_classes)
#                loc shape: torch.size(batch_size,num_priors,4)
#                priors shape: torch.size(num_priors,4)
#
#            targets (tensor): Ground truth boxes and labels for a batch,
#                shape: [batch_size,num_objs,5] (last idx is the label).
# =============================================================================
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            if self.use_gpu:
                defaults = defaults.cuda()
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # filter out pos boxes for now
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        #N = num_pos.data.sum()
        #loss_l /= N
        #loss_c /= N

        N = num_pos.data.sum().double()
        loss_l = loss_l.double()
        loss_c = loss_c.double()   
        return loss_l, loss_c

"""
print(NET_CRITERION)

# Define the TRANSFORM functions
NET_TRANSFORM = """

def detection_collate(batch):
# ======================================================================================
#    Custom collate fn for dealing with batches of images that have a different
#    number of associated object annotations (bounding boxes).
#
#    Arguments:
#        batch: (tuple) A tuple of tensor images and lists of annotations
#
#    Return:
#        A tuple containing:
#            1) (tensor) batch of images stacked on their 0 dim
#            2) (list of tensors) annotations for a given image are stacked on
#                                 0 dim
# ======================================================================================
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets
    
def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]

def jaccard_numpy(box_a, box_b):
# ======================================================================================
#    Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
#    is simply the intersection over union of two boxes.
#    E.g.:
#        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
#    Args:
#        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
#        box_b: Single bounding box, Shape: [4]
#    Return:
#        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
# ======================================================================================
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
# ======================================================================================
#    Composes several augmentations together.
#    Args:
#        transforms (List[Transform]): list of transforms to compose.
#    Example:
#        >>> augmentations.Compose([
#        >>>     transforms.CenterCrop(10),
#        >>>     transforms.ToTensor(),
#        >>> ])
# ======================================================================================

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
# ======================================================================================
#    Applies a lambda as a transform.
# ======================================================================================

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size,
                                 self.size))
        return image, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class RandomSampleCrop(object):
# ======================================================================================
#    Crop
#    Arguments:
#        img (Image): the image being input during training
#        boxes (Tensor): the original bounding boxes in pt form
#        labels (Tensor): the class labels for each bbox
#        mode (float tuple): the min and max jaccard overlaps
#    Return:
#        (img, boxes, classes)
#            img (Image): the cropped image
#            boxes (Tensor): the adjusted bounding boxes in pt form
#            labels (Tensor): the class labels for each bbox
# ======================================================================================
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class SwapChannels(object):
# ======================================================================================
#    Transforms a tensorized image by swapping the channels in the order
#     specified in the swap tuple.
#    Args:
#        swaps (int triple): final order of channels
#            eg: (2, 1, 0)
# ======================================================================================

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
# ======================================================================================
#        Args:
#            image (Tensor): image tensor to be transformed
#        Return:
#            a tensor with channels swapped according to swap
# ======================================================================================
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)


class SSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)
"""
print(NET_TRANSFORM)


if 'variables' in locals():
  variables.put("NET_MODEL", NET_MODEL)
  variables.put("NET_CRITERION", NET_CRITERION)
  variables.put("NET_TRANSFORM", NET_TRANSFORM)    
  variables.put("IMG_SIZE", IMG_SIZE)
  variables.put("START_ITERATION", START_ITERATION)
  variables.put("MAX_ITERATION", MAX_ITERATION)
  variables.put("LR_STEPS", LR_STEPS)
  variables.put("LR_FACTOR", LR_FACTOR)
  variables.put("GAMMA", GAMMA)
  variables.put("USE_PRETRAINED_MODEL", USE_PRETRAINED_MODEL)
  variables.put("NUM_CLASSES", NUM_CLASSES)
  variables.put("MIN_SIZES", MIN_SIZES)
  variables.put("MAX_SIZES", MAX_SIZES)
  variables.put("LEARNING_RATE",  LEARNING_RATE) 
  variables.put("MOMENTUM",  MOMENTUM) 
  variables.put("WEIGHT_DECAY", WEIGHT_DECAY)
  variables.put("LABEL_PATH", LABEL_PATH)
  variables.put("NET_NAME", NET_NAME)

print("END SSD")