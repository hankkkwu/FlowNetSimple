# Download dataset
# !wget https://thinkautonomous-flownet.s3.eu-west-3.amazonaws.com/flownet-data.zip


"""Imports"""
# from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import os.path
import os
from imageio import imread
import numbers
from pathlib import Path
import shutil
import random
import time
from tqdm import tqdm
import torch.utils.data as data
import torch.utils.data as data
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from torch.utils.tensorboard import SummaryWriter
import flow_transforms  # this is provided by us
from read_kitti import read_png_file, flow_to_image


def bgr2rgb(image):
    """
    Convert BGR TO RGB
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def train_test_split(images, default_split=0.8):
    """
    Splits the Dataset Paths into Train/Test. A good ratio would be 80% training and 20% testing.
    """
    split_values = np.random.uniform(0,1,len(images)) < default_split # Randomly decides if an image is train or test
    train_samples = [sample for sample, split in zip(images, split_values) if split]
    test_samples = [sample for sample, split in zip(images, split_values) if not split]
    return train_samples, test_samples


def load_flow_from_png(png_path):
    '''
    This is used to read flow label images from the KITTI Dataset
    '''
    # The Image is a 16 Bit(uint16) Image. We must read it with OpenCV and
    # the flag cv2.IMREAD_UNCHANGED (-1)

    # The first channel denotes if the pixel is valid or not (1 if true, 0 otherwise),
    # the second channel contains the v-component and the third channel the u-component.
    flo_file = cv2.imread(png_path, -1)   # (375, 1242, 3)
    flo_img = flo_file[:,:,2:0:-1].astype(np.float32)   # (375, 1242, 2)

    # See the README File in the KITTI DEVKIT AND THE FLOW READER FUNCTIONS
    # To convert the u-/v-flow into floating point values, convert the value
    # to float, subtract 2^15(32768) and divide the result by 64.0
    invalid = (flo_file[:,:,0] == 0)
    flo_img = flo_img - 32768
    flo_img = flo_img / 64

    # Valid and Small Flow = 1e-10
    flo_img[np.abs(flo_img) < 1e-10] = 1e-10

    # Invalid Flow = 0
    flo_img[invalid, :] = 0
    return flo_img

def KITTI_loader(root, path_imgs, path_flo):
    """
    Returns the Loaded Images in RGB, and the Loaded Optical Flow Labels.
    NOTE: KITTI dataset has sparse optical flow ground truth
    """
    #TODO: Implement the function and return [[img1, img2], flow_image]
    imgs = [os.path.join(root, path) for path in path_imgs]
    flo = os.path.join(root, path_flo)
    # img[:,:,::-1] will do: bgr -> rgb or rgb -> bgr
    return [cv2.imread(img)[:,:,::-1].astype(np.float32) for img in imgs], load_flow_from_png(flo)

class ListDataset(data.Dataset):
    def __init__(self, path_list, transform=None, target_transform=None, co_transform=None, loader=KITTI_loader):
        self.root = os.getcwd()
        self.path_list = path_list
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        In Python, __getitem__ is used to read values from a class. For example; read the transformed input files.
        Instead of calling the function .read(), we use __getitem__ to directly get the value.
        Similarly, __setitem__ can be used to fill values in a class.
        """
        inputs, target = self.path_list[index]
        inputs, target = self.loader(self.root, inputs, target)
        if self.co_transform is not None:
            inputs, target = self.co_transform(inputs, target)
        if self.transform is not None:
            inputs[0] = self.transform(inputs[0])
            inputs[1] = self.transform(inputs[1])
        if self.target_transform is not None:
            target = self.target_transform(target)
        return inputs, target

    def __len__(self):
        return len(self.path_list)


"""
PART I - Load and Prepare the Dataset for the Model
"""
# load the dataset
images_dataset = sorted(glob.glob("dataset/images_2/*.png"))
labels_dataset = sorted(glob.glob("dataset/flow_occ/*.png"))

# Make a List of images [[[Img1, Img2], Flow Map], ...]
images = []
for flow_map in labels_dataset:
    root_filename = flow_map[-13:-7]
    img1 = os.path.join("dataset/images_2/", root_filename+'_10.png')   # training images origianl shape = 1242 x 375
    img2 = os.path.join("dataset/images_2/", root_filename+'_11.png')
    images.append([[img1, img2], flow_map])

# Call Train/Test split function
train_samples, test_samples = train_test_split(images)
# print("num of training samples: ", len(train_samples))
# print("num of test samples: ", len(test_samples))

# Do image transformations
div_flow = 20   #Factor by which we divide the output (thus >=1). It makes training more stable to deal with low numbers than big ones.
input_transform = transforms.Compose([flow_transforms.ArrayToTensor(), transforms.Normalize(mean=[0,0,0], std=[255,255,255]), transforms.Normalize(mean=[0.45,0.432,0.411], std=[1,1,1])])
target_transform = transforms.Compose([flow_transforms.ArrayToTensor(),transforms.Normalize(mean=[0,0],std=[div_flow,div_flow])])
co_transform = flow_transforms.Compose([flow_transforms.RandomCrop((320,448)), flow_transforms.RandomVerticalFlip(),flow_transforms.RandomHorizontalFlip()])    # make training img shape to be 320 x 448

# Get train and test dataset
train_dataset = ListDataset(train_samples, input_transform, target_transform, co_transform, loader=KITTI_loader)
test_dataset = ListDataset(test_samples, input_transform, target_transform, flow_transforms.CenterCrop((370,1224)), loader=KITTI_loader)



"""
PartII - Build a FlowNet Architecture
"""

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    """
    Define Convolution with LeakyReLU, and with or without batchNorm for Encoder
    """
    if batchNorm:
        # convolution in 2D with Batchnorm and leakyReLU of 0.1
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        # Convolution in 2D with LeakyReLU of 0.1 and without Batchnorm
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
        )

def predict_flow(in_planes):
    """
    predict the output(flow map)
    """
    # The depth of the flow map is 2, which contains (u, v)
    # NOTE: The kernel size in the paper is 5, but we need 3 to make the dimension(width and height) right
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=False)

def upsampling_flow():
    """
    Upsampling the predicted flow for concatenating
    """
    return nn.ConvTranspose2d(2, 2, 4, stride=2, padding=1, bias=False)

def deconv(in_planes, out_planes):
    """
    Define deconvolution for upsampling
    """
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1, inplace=True)
    )

def crop_like(input, target):
    """
    crop the input width and height to be the same as target
    input shape = [B, C, H, W]
    target shape = [B, C, H, W]
    """
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]


class FlowNetS(nn.Module):
    """
    Build FlownetS(Simple), which is a simple version using 2D Convolutions to get to the optical flow computation
    """
    expansion = 1

    def __init__(self, batchNorm=True):
        super(FlowNetS, self).__init__()

        # Encoder part
        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, 6, 64, kernel_size=7, stride = 2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride = 2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride = 2)
        self.conv3_1 = conv(self.batchNorm, 256, 256, kernel_size=3, stride = 1)
        self.conv4 = conv(self.batchNorm, 256, 512, kernel_size=3, stride = 2)
        self.conv4_1 = conv(self.batchNorm, 512, 512, kernel_size=3, stride = 1)
        self.conv5 = conv(self.batchNorm, 512, 512, kernel_size=3, stride = 2)
        self.conv5_1 = conv(self.batchNorm, 512, 512, kernel_size=3, stride = 1)
        self.conv6 = conv(self.batchNorm, 512, 1024, kernel_size=3, stride = 2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024, kernel_size=3, stride = 1)

        # Refinement part
        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = upsampling_flow()
        self.upsampled_flow5_to_4 = upsampling_flow()
        self.upsampled_flow4_to_3 = upsampling_flow()
        self.upsampled_flow3_to_2 = upsampling_flow()

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        for m in self.modules():
            """
            The self.modules() method returns an iterable to the many layers or “modules” defined in the model class.
            This is using that self.modules() iterable to initialize the weights of the different layers present in the model.

            isinstance() checks if the particular layer “m” is an instance of a conv2d or ConvTranspose2d or BatchNorm2d layer etc.
            and initializes the weights accordingly.
            """
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # Initialize the Convolutions with "He Initialization" to 0.1 (https://arxiv.org/pdf/1502.01852.pdf)
                kaiming_normal_(m.weight, 0.1)  # 0.1 is the negative slope of the rectifier used after this layer (only used with 'leaky_relu')
                if m.bias is not None:
                    # Initialize all bias to 0
                    constant_(m.bias, 0)
            # Initialize the BatchNorm Convolutions
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        # Encoder part
        # x = [b, 6, 320, 448]
        conv1 = self.conv1(x)           # b x 64 x 160 x 224
        conv2 = self.conv2(conv1)       # b x 128 x 80 x 112
        conv3 = self.conv3(conv2)       # b x 256 x 40 x 56
        conv3_1 = self.conv3_1(conv3)   # b x 256 x 40 x 56
        conv4 = self.conv4(conv3_1)     # b x 512 x 20 x 28
        conv4_1 = self.conv4_1(conv4)   # b x 512 x 20 x 28
        conv5 = self.conv5(conv4_1)     # b x 512 x 10 x 14
        conv5_1 = self.conv5_1(conv5)   # b x 512 x 10 x 14
        conv6 = self.conv6(conv5_1)     # b x 1024 x 5 x 7
        conv6_1 = self.conv6_1(conv6)   # b x 1024 x 5 x 7

        # Refinement part
        flow6 = self.predict_flow6(conv6_1)
        flow6_upsampling = self.upsampled_flow6_to_5(flow6)
        flow6_upsampling = crop_like(flow6_upsampling, conv5_1)

        deconv5 = self.deconv5(conv6_1)
        deconv5 = crop_like(deconv5, conv5_1)
        concat5 = torch.cat((conv5_1, deconv5, flow6_upsampling), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_upsampling = self.upsampled_flow5_to_4(flow5)
        flow5_upsampling = crop_like(flow5_upsampling, conv4_1)

        deconv4 = self.deconv4(concat5)
        deconv4 = crop_like(deconv4, conv4_1)
        concat4 = torch.cat((conv4_1, deconv4, flow5_upsampling), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_upsampling = self.upsampled_flow4_to_3(flow4)
        flow4_upsampling = crop_like(flow4_upsampling, conv3_1)

        deconv3 = self.deconv3(concat4)
        deconv3 = crop_like(deconv3, conv3_1)
        concat3 = torch.cat((conv3_1, deconv3, flow4_upsampling), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_upsampling = self.upsampled_flow3_to_2(flow3)
        flow3_upsampling = crop_like(flow3_upsampling, conv2)

        deconv2 = self.deconv2(concat3)
        deconv2 = crop_like(deconv2, conv2)
        concat2 = torch.cat((conv2, deconv2, flow3_upsampling), 1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def flownets(data=None, batchNorm=False):
    """
    FlowNetS model architecture from the paper (https://arxiv.org/abs/1504.06852)
    Args:
    data: pretrained weights of the network
    """
    model = FlowNetS(batchNorm=batchNorm)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


model_to_load = "models/flownets_bn_EPE2.459.pth.tar"
# if using CPU, second parameter: map_location=torch.device("cpu")
# because the pretrained model was trained using GPU, it will cause problem when using cpu for training
checkpoint = torch.load(model_to_load, map_location=torch.device("cpu"))
model = flownets(data=checkpoint, batchNorm=True)
# print(model)



"""
PartIII: Train the model on KITTI
"""
########################################
# define hyperparameters and variables #
########################################
arch = "flownetsbn"
solver = "adam"
epochs = 200
epoch_size = 0
batch_size = 64
learning_rate = 10e-4
workers = 4             # how many subprocesses to use for data loading.
pretrained = None
bias_decay = 0
weight_decay = 4e-4
momentum = 0.9
milestones = [100,150,200]  # epochs by which we divide learning rate by 2

save_path = '{}_{}_{}epochs_b{}_lr{}'.format(arch, solver, epochs, batch_size, learning_rate)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# When an optimizer is instantiated, parameter group is the as well as a variety of hyperparameters such as the learning rate.
# Optimizers are also passed other hyperparameters specific to each optimization algorithm. It can be extremely useful to set up
# groups of these hyperparameters, which can be applied to different parts of the model. This can be achieved by creating a
# parameter group, essentially a list of dictionaries that can be passed to the optimizer.

# The param variable must either be an iterator over a torch.tensor or a Python dictionary specifying a default value of optimization options.
# Note that the parameters themselves need to be specified as an ordered collection, such as a list, so that parameters are a consistent sequence
param_groups = [{'params': model.bias_parameters(), 'weight_decay': bias_decay},
                {'params': model.weight_parameters(), 'weight_decay': weight_decay}]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

if solver == "adam":
    optimizer = torch.optim.Adam(param_groups, learning_rate)
elif solver == "sgd":
    optimizer = torch.optim.SGD(param_groups, learning_rate, momentum=momentum)

# using writers to plug values or for tensorboard visualization
# Writes entries directly to event files in the log_dir to be consumed by TensorBoard.
train_writer = SummaryWriter(log_dir=os.path.join(save_path, "train"))    # from torch.utils.tensorboard import SummaryWriter
test_writer = SummaryWriter(log_dir=os.path.join(save_path, "trst"))
output_writers = []
for i in range(3):
    output_writers.append(SummaryWriter(os.path.join(save_path, "test", str(i))))


# Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=workers, pin_memory=True, shuffle=True)
val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=workers, pin_memory=True, shuffle=False)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)


#########################################################
# Define the Loss Function as the End Point Error (EPE) #
#########################################################
def EPE(input_flow, target_flow, sparse=False, mean=True):
    """
    Define End Point Error (EPE)
    Args:
    input_flow and target_flow have the same shape [b,c,h,w]
    """
    EPE_map = torch.norm(target_flow - input_flow, p=2, dim=1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0]==0) & (target_flow[:,1]==0)
        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()/batch_size

def realEPE(predicted_flow, target, sparse=False):
    """
    Since the prediction is not the same size as the ground truth,
    we need to resize it to the same as the ground truth, and then calculate the EPE
    Args:
    predicted_flow: shape = [b,c,h,w]
    target: shape = [b,c,h,w]
    """
    b, _, h, w = target.size()
    upsampled_output = F.interpolate(predicted_flow, (h,w), mode="bilinear", align_corners=False)   # used to resize the output
    return EPE(upsampled_output, target, sparse, mean=True)

def sparse_max_pool(target, size):
    """
    Downsample the target by considering 0 values as invalid.
    Unfortunately, no generic interpolation mode can resize a sparse map correctly,
    the strategy here is to use max pooling for positive values and "min pooling"
    for negative values, the two results are then summed.
    This technique allows sparsity to be minized, contrary to nearest interpolation,
    which could potentially lose information for isolated data points.
    Args:
    target: the ground truth flow map.
    size: the size we want the target to be resized to.
    """
    positive = (target > 0).float()
    negative = (target < 0).float()
    output = F.adaptive_max_pool2d(target * positive, size) - F.adaptive_max_pool2d(-target * negative, size)
    return output


def multiscaleEPE(network_output, target_flow, weights=None, sparse=False):
    """
    compute the loss (sum of all 5 flow map's EPE)
    Args:
    network_output: list of predicted flow (flow2, flow3, flow4, flow5, flow6)
    target_flow: ground truth flow map
    """
    def one_scale(predicted_flow, target, sparse):
        """
        resize the ground truth flow map to the same size as the predicted_flow, so we can calculate the loss
        """
        b, _, h, w = predicted_flow.size()
        if sparse:
            target_scaled = sparse_max_pool(target, (h,w))
        else:
            target_scaled = F.interpolate(target, (h, w), mode="area")
        return EPE(predicted_flow, target_scaled, sparse, mean=False)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]   # as in original article
    assert(len(weights) == len(network_output))

    # compute the loss of each predicted flow map
    loss = 0
    for flow, weight in zip(network_output, weights):
        loss += weight * one_scale(flow, target_flow, sparse)
    return loss


##########################################
# Create some useful class and functions #
##########################################
class AverageMeter(object):
    """
    Computes and stores the average and current value.
    It's quite useful in our case where we have to average a loss over several pixels and several frames.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n    # not sure why * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def save_checkpoint(state, is_best, save_path, filename="checkpoint.pth.tar"):
    """
    Save a checkpoint to continue training from the last checkpoint
    """
    torch.save(state, os.path.join(save_path, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))


def flow2rgb(flow_map, max_value):
    """
    Used to visualize the output after a forward pass
    https://github.com/ClementPinard/FlowNetPytorch/issues/86
    """
    flow_map_np = flow_map.detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3,h,w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0,1)


###############################################
# Define train function to train on one epoch #
###############################################
def train(train_loader, model, optimizer, epoch, train_writer):
    """
    Train on one epoch
    Args:
    train_loader: contains [images, labels], images shape = [2, batch_size, channel, height, width], label = [batch_size, 2(u,v), height, width]
    """
    global n_iter, div_flow
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    flow2_EPEs = AverageMeter()

    multiscale_weights = [0.005, 0.01, 0.02, 0.08, 0.32]
    epoch_size = len(train_loader)

    # train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):  # len(train_loader) = 3
        # go through the entire data loader

        # update the data_time
        data_time.update(time.time()-end)

        input = torch.cat(input, 1).to(device)  # concatenete 2 images(t = 0 and t = 1), shape = [64, 6, 320, 448]
        target = target.to(device)

        # forward pass
        output = model(input)   # output shape: [predicted flow maps, batch_size, channel, height, width]

        # Since Target pooling is not very precise when sparse, take the highest
        # resolution prediction and upsample it instead of downsampling target.
        h, w = target.size()[-2:]
        output = [F.interpolate(output[0], (h,w)), *output[1:]]     # output[0] is the predicted flow2, * is used to unpack the output from flow3 to flow6

        # Compute Multiscale EPE (for all predict flows)
        # NOTE: KITTI dataset has sparse optical flow ground truth
        loss = multiscaleEPE(output, target, weights=multiscale_weights, sparse=True)

        # Compute the output(flow2) EPE
        # Since we divided the output by div_flow at the beginning, we need to multiply it back
        flow2_EPE = div_flow * realEPE(output[0], target, sparse=True)

        # Record loss and flow2 EPE
        losses.update(loss.item(), target.size(0))  # target.size(0) = batch_size
        train_writer.add_scalar('train_loss', loss.item(), n_iter)   # Add scalar data to summary.
        flow2_EPEs.update(flow2_EPE.item(), target.size(0))

        # backward pass: compute the gradient and do optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # scheduler will divide the learning rate by 2 at epoch [100, 150, 200]
        scheduler.step()

        # measure the elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print the loss and EPE
        print('Epoch: [{}][{}/{}]\t Time {}\t Data_time {}\t Loss {}\t EPE {}'.format(
                epoch, i, epoch_size, batch_time, data_time, losses, flow2_EPEs))

        n_iter += 1
        if i >= epoch_size:
            break

    # return the average loss and average EPE of flow2 on training set
    return losses.avg, flow2_EPEs.avg


############################################
# define validation function for one epoch #
############################################
def validate(val_loader, model, epoch, output_writers):
    """
    Validate on one epoch
    """
    global div_flow
    batch_time = AverageMeter()
    flow2_EPEs = AverageMeter()

    # evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):    # len(val_loader) = 1
        # go through the entire validation loader
        input = torch.cat(input, 1).to(device)
        target = target.to(device)

        # forward pass
        output = model(input)   # output shape: [batch_size, channel, height, width]

        # compute EPE
        flow2_EPE = div_flow * realEPE(output, target, sparse=True)

        # Record EPE
        flow2_EPEs.update(flow2_EPE.item(), target.size(0))     # target.size(0) = batch_size

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # log first output of the first batches
        if i < len(output_writers): # len(output_writers) = 3
            if epoch == 0:
                mean_values = torch.tensor([0.45, 0.432, 0.411], dtype=input.dtype).view(3,1,1)
                # add_image: Add image data to summary. Note that this requires the pillow package.
                # add_image(tag, img_tensor, global_step=None)
                # img_tensor(torch.Tensor, numpy.array, or string/blobname) – Image data
                # global_step(int) – Global step value to record
                output_writers[i].add_image('GroundTruth', flow2rgb(div_flow * target[0], max_value=10), 0)
                output_writers[i].add_image('Inputs', (input[0, :3].cpu() + mean_values).clamp(0,1), 0)
                output_writers[i].add_image('Inputs', (input[0, :3].cpu() + mean_values).clamp(0,1), 1)
            output_writers[i].add_image('FlowNet Outputs', flow2rgb(div_flow * output[0], max_value=10), epoch)

        # print predicted flow map EPE
        print('Test: [{0}/{1}]\t Time {2}\t EPE {3}'.format(i, len(val_loader), batch_time, flow2_EPEs))
    print(' * EPE {:.3f}'.format(flow2_EPEs.avg))
    return flow2_EPEs.avg



#####################################
# Main - for training and validatin #
#####################################
def main():
    save_path = '{}_{}_{}epochs{}_b{}_lr{}'.format(arch, solver, epochs, '_epochSize'+str(epoch_size) if epoch_size > 0 else '', batch_size, learning_rate)
    n_iter = 0
    best_EPE = -1

    # start from a model that is pretrained on "Flying Chairs" and finetune it to KITTI
    save_path = os.path.join("models", save_path)
    print("checkpoints will be saved to {}".format(save_path))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for epoch in range(epochs):
        # Train for one epoch
        train_loss, train_EPE = train(train_loader, model, optimizer, epoch, train_writer)
        train_writer.add_scalar("mean EPE", train_EPE, epoch)   # Add scalar data to summary.

        # Evaluate on validation set
        with torch.no_grad():
            endpointerror = validate(val_loader, model, epoch, output_writers)
        test_writer.add_scalar('mean EPE', endpointerror, epoch)

        # store the best EPE
        if best_EPE < 0:
            best_EPE = endpointerror
        is_best = endpointerror < best_EPE    # if EPE < best_EPE, then the EPE is the best, is_best = Ture
        best_EPE = min(endpointerror, best_EPE)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': arch,
            'state_dict': model.module.state_dict(),
            'best_EPE': best_EPE,
            'div_flow': div_flow
            }, is_best, save_path)


"""
Part IV - Inference
"""
###############
# On 2 images #
###############
input_transform = transform.Compose([flow_transforms.ArrayToTensor(),
                                     transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
                                     transfomrs.Normalize(mean=[0.411,0.4332, 0.45], std=[1,1,1])])
network_data = torch.load("model_beset.path.tar")
div_flow = network_data['div_flow']

model = flownet(network_data, batchNorm=True).to(device)
model.eval()

# 在一開始時花費一點額外的時間，為整個neural net的每個conv layer做搜索，
# 找出最適合他的convlution算法，進而實踐網路的加速。適用的場景是：固定的
# 網路結構及輸入的參數(如batch size, image size, input channel)
cudnn.benchmark=True

# randomly choose pics
idx = random.randint(0,len(train_samples))
img1_file = train_samples[idx][0][0]
img2_file = train_samples[idx][0][1]
flow_target = flow_to_image(read_png_file(train_samples[idx][1]))

with torch.no_grad():
    img1 = input_transform(imread(img1_file))       # from imageio import imread
    img2 = input_transform(imread(img2_file))       # try to use cv2.imread(img, -1)
    # input_var = torch.cat([img1, img2]).unsqueeze(0)
    # input_var = input_var.to(device)
    input_var = torch.cat([img1, img2], 1).to(device)
    output = model(input_var)

    for suffix, flow_output in zip(['flow', 'inv_flow'], output):
        filename = img1_file[:-4] + "flow"
        rgb_flow = flow2rgb(div_flow * flow_output, max_value=None)
        rgb_flow = (rgb_flow * 255).astype(np.uint8).transpose(1,2,0)

f, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(30,20))
ax0.imshow(bgr2rgb(cv2.imread(img1_file)))
ax0.set_title("Original Image", fontsize=30)
ax1.imshow(rgb_flow)
ax1.set_title('Prediction', fontsize=30)
ax2.imshow(flow_target)
ax2.set_title('Ground Truth', fontsize=30)



##############
# On a video #
##############
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

video_idx = 2
video_images = sorted(glob.glob("videos/video"+str(video_idx)+"/*.png"))
result_video = []
im1 = None

for idx_run, img in enumerate(video_images):
    if idx_run == 0:
        im1 = imread(img)
        idx_run += 1
    else:
        im2 = imread(img)
        with torch.no_grad():
            img1 = input_transform(im1)
            img2 = imput_transform(im2)
            input_var = torch.cat([img1, img2], 1).to(device)
            output = model(input_var)

            for suffix, flow_output in zip(['flow', 'inv_flow'], output):
                rgb_flow = flow2rgb(div_flow * flow_output, max_value=None)
                rgb_flow = (rgb_flow * 255).astype(np.uint8).transpose(1,2,0)
                result_video.append(cv2.cvtColor(rgb_flow, cv2.COLOR_RGB2BGR))

out = cv2.VideoWriter("output/out-"+str(video_idx)+".mp4",cv2.VideoWriter_fourcc(*'MP4V'), 15.0, (311 ,94))
for i in range(len(result_video)):
    out.write(result_video[i])
out.release()
