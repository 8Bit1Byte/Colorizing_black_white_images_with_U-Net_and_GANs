import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gs
from matplotlib import font_manager as fm, rcParams
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

from tqdm import tqdm

from datetime import datetime
from helper_methods import *
from generative_model import *
from discriminator_model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import fastai

# Download Dataset
from fastai.data.external import untar_data, URLs
coco_path = untar_data(URLs.COCO_SAMPLE)
coco_path = str(coco_path) + "/train_sample"
paths = glob.glob(coco_path + "/*.jpg")
# Setting seed for getting the same data across all train sessions 
np.random.seed(123)
paths_subset = np.random.choice(paths, 10_000, replace=False) # choosing 10000 images randomly
rand_idxs = np.random.permutation(10_000)
train_idxs = rand_idxs[:8000] # choosing the first 8000 as training set
val_idxs = rand_idxs[8000:] # choosing last 2000 as validation set
train_paths = paths_subset[train_idxs]
val_paths = paths_subset[val_idxs]
print(train_paths)

ImageSize = 256
class MakeDataset(Dataset):
    def __init__(self, paths):
        self.transforms = transforms.Compose([
                transforms.Resize((ImageSize, ImageSize),  transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(), # Added after 350th Epoch to see if Results improves
            ])
        self.paths=paths

    def __getitem__(self, i):
        img = Image.open(self.paths[i])
        img = img.convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        imgInLAB = rgb2lab(img).astype("float32")
        imgInLAB = transforms.ToTensor()(imgInLAB)
        L_array = imgInLAB[[0], ...] / 50. - 1.
        ab_array = imgInLAB[[1, 2], ...] / 110.
        return [L_array, ab_array]
        
    def __len__(self):
        return len(self.paths)
    
BatchSize, Workers = [16, 4]
trainDL = DataLoader(MakeDataset(paths=train_paths), batch_size=BatchSize, num_workers=Workers, pin_memory=True, shuffle = True)
validationDL = DataLoader(MakeDataset(paths=val_paths), batch_size=BatchSize, num_workers=Workers, pin_memory=True, shuffle = True)


def lab_to_rgb(L, ab):  
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

data = next(iter(trainDL))
L_Array, ab_Array = data[0], data[1]
print(f"L Array Shape : {L_Array.shape}", f"*a*b Array Shape : {ab_Array.shape}",sep='\n')

fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(15,10))
ax0.imshow(L_Array[0][0], cmap='gray')
ax0.set_title('L')
ax1.imshow(ab_Array[0][0])
ax1.set_title('a')
ax2.imshow(ab_Array[0][1])
ax2.set_title('b')
ax3.imshow(lab_to_rgb(L_Array,ab_Array)[0])
ax3.set_title('RGB')
plt.show()


# Initializing The Model
# Defining Some Hyperparameters
LEARNING_RATE = 2e-4
EPOCHS = 950
LAMBDA = 100 #Discriminator L1 Loss Hyperparameter as Defined in the Pix2Pix Paper 
epoch = 1
BETAS = (0.5,0.999) #Optimizer Hyperparameter as Defined in the Pix2Pix Paper
lossOfDiscriminator = []
lossOfGenerator = []

# Functions and Logic for Loading and Saving Checkpoints
inputFolder = "../input/model-params"
outputFolder = "/kaggle/working"
checkpointPathDiscriminator = inputFolder+"/disc.pth.tar"
checkpointPathGenerator = inputFolder+"/gen.pth.tar"
loadModel = True
def SaveCheckpoint(model, optimizer, epoch, filename):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch":epoch,
        "DISC_LOSS" : lossOfDiscriminator,
        "GEN_LOSS" : lossOfGenerator
    }
    torch.save(checkpoint, filename)

def LoadCheckpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    global epoch
    global lossOfDiscriminator
    global lossOfGenerator
    epoch = checkpoint["epoch"]
    lossOfDiscriminator = checkpoint["DISC_LOSS"].copy()
    lossOfGenerator = checkpoint["GEN_LOSS"].copy()

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

# Initializing Models
discModel = Discriminator(3).to(device)
genModel = Generator(1).to(device)
optimizerForDiscriminator = optim.Adam(discModel.parameters(),lr=LEARNING_RATE, betas=BETAS)
optimizerForGenerator = optim.Adam(genModel.parameters(),lr=LEARNING_RATE, betas=BETAS)
LossFunction = nn.BCEWithLogitsLoss()
L1Loss = nn.L1Loss()
#Float 16 Training for faster Training
discriminatorScaler = torch.cuda.amp.GradScaler()
generatorScaler = torch.cuda.amp.GradScaler()

# Loading Previously Saved Checkpoint if Applicable
if loadModel:
    LoadCheckpoint(checkpointPathGenerator, genModel, optimizerForGenerator, LEARNING_RATE)
    LoadCheckpoint(checkpointPathDiscriminator, discModel, optimizerForDiscriminator, LEARNING_RATE)
SaveModel = True
checkpointPathDiscriminator = outputFolder+"/disc.pth.tar"
checkpointPathGenerator = outputFolder+"/gen.pth.tar"

# Training
def TrainFunction(discModel, genModel, loader, optimizerForDiscriminator, optimizerForGenerator, L1Loss, BCELoss, generatorScaler, discriminatorScaler):
    loop = tqdm(loader, leave=True)
    for idx, (L, ab) in enumerate(loop):
        L = L.to(device)
        ab = ab.to(device)
        
        # Train Discriminator
        with torch.cuda.amp.autocast():
            YGenerated = genModel(L)
            discReal = discModel(torch.concat([L, ab],1))
            discRealLoss = BCELoss(discReal, torch.ones_like(discReal))
            discGenerated = discModel(torch.concat([L, YGenerated.detach()],1))
            discGeneratedLoss = BCELoss(discGenerated, torch.zeros_like(discGenerated))
            discriminatorLoss = (discRealLoss + discGeneratedLoss) / 2
            lossOfDiscriminator.append(discriminatorLoss.item())
        discModel.zero_grad()
        discriminatorScaler.scale(discriminatorLoss).backward()
        discriminatorScaler.step(optimizerForDiscriminator)
        discriminatorScaler.update()
        
        # Train generator
        with torch.cuda.amp.autocast():
            discGenerated = discModel(torch.concat([L, YGenerated],1))
            genGeneratedLoss = BCELoss(discGenerated, torch.ones_like(discGenerated))
            L1 = L1Loss(YGenerated, ab) * LAMBDA
            generatorLoss = genGeneratedLoss + L1
            lossOfGenerator.append(generatorLoss.item())

        optimizerForGenerator.zero_grad()
        generatorScaler.scale(generatorLoss).backward()
        generatorScaler.step(optimizerForGenerator)
        generatorScaler.update()
TRAIN=True 
visualizeWhileTraining=True
saveImages = True #To Save Images during visualization

while TRAIN and (epoch <= EPOCHS):
    print("\nEpoch",epoch,'\n')
    
    if visualizeWhileTraining:
        ShowSamples(genModel, validationDL,outputFolder,epoch,saveImages)
        
        print("Generator Loss\n")
        VisualizeLoss(lossOfGenerator,outputFolder,epoch,True,saveImages)
        print("Discriminator Loss\n")
        VisualizeLoss(lossOfDiscriminator,outputFolder,epoch,False,saveImages)
        
    if SaveModel:
        SaveCheckpoint(genModel, optimizerForGenerator, epoch, filename=checkpointPathGenerator)
        SaveCheckpoint(discModel, optimizerForDiscriminator, epoch, filename=checkpointPathDiscriminator)

    TrainFunction(discModel, genModel, trainDL, optimizerForDiscriminator, optimizerForGenerator, L1Loss, LossFunction, discriminatorScaler, generatorScaler)
    
    epoch+=1

# Step 8. Visualizing Loss Trajectory
# Generator Loss - Average Loss
VisualizeAvgLoss(lossArr=lossOfGenerator,folder=outputFolder,epoch=epoch,generator=True,SAVE=True,windowSize=100)
# Generator Loss - Actual Loss
VisualizeLoss(lossArr=lossOfGenerator,folder=outputFolder,epoch=epoch,generator=True,SAVE=True)
# Discriminator Loss - Average Loss
VisualizeAvgLoss(lossArr=lossOfDiscriminator,folder=outputFolder,epoch=epoch,generator=False,SAVE=True,windowSize=1000)
# Discriminator Loss - Actual Loss
VisualizeLoss(lossArr=lossOfDiscriminator,folder=outputFolder,epoch=epoch,generator=False,SAVE=True)

# Visualizing Predictions
# Predictions on Training Data
numRuns = 200 #Generate numRuns*5 Samples
for run in range(numRuns):
    ShowSamples(genModel, trainDL,outputFolder,epoch,SAVE=True,suffix="_On_Training_Set")

# Predictions on Validation Data
for run in range(numRuns):
    ShowSamples(genModel, validationDL,outputFolder,epoch,SAVE=True, suffix="_On_Validation_set")