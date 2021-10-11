import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def ShowSamples(Model, dl, folder='./', epoch=-1, SAVE=True, suffix=""):
    data = next(iter(dl))
    L, ab = data[0], data[1]
    L = L.to(device)
    ab = ab.to(device)
    # Setting Model to Evaluation Mode. This disables layers like dropout
    Model.eval()
    with torch.no_grad():
        abGenerated = Model(L)
    Model.train()
    inputImages = lab_to_rgb(L, ab)
    generatedImages = lab_to_rgb(L, abGenerated)
    # Row = Number of samples generated per run (Keep it smaller than ${BatchSize}, Col=constant, img = Image size.
    row, col, img = 1, 3, 5
    fig = plt.figure(figsize=(col*img, row*img))
    gs1 = gs.GridSpec(nrows=row, ncols=col)
    for i in range(row):
        ax = plt.subplot(gs1[i, 0])
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax.set_title('Grayscale', fontsize=16, fontweight='bold')
        ax = plt.subplot(gs1[i, 1])
        ax.imshow(generatedImages[i])
        ax.axis("off")
        ax.set_title('Prediction', fontsize=16, fontweight='bold')
        ax = plt.subplot(gs1[i, 2])
        ax.imshow(inputImages[i])
        ax.axis("off")
        ax.set_title('Ground Truth', fontsize=16, fontweight='bold')

    plt.subplots_adjust(wspace=0, hspace=0.1)
    plt.show()

    if SAVE:
        now = datetime.now()
        currentTime = now.strftime("%H:%M:%S")
        fig.savefig(
            folder + f"/Results_After_Epoch_{epoch}{suffix}_{currentTime}.png")


def VisualizeLoss(lossArr, folder, epoch, generator=True, SAVE=True):
    x = (range(0, len(lossArr)))
    plt.figure(figsize=(12, 10))
    plt.plot(x, lossArr)
    str = "Discriminator"
    if generator:
        str = "Generator"

    plt.xlabel("Number of Iterations")
    plt.ylabel(str + " Loss")
    if SAVE:
        now = datetime.now()
        currentTime = now.strftime("%H:%M:%S")
        plt.savefig(
            folder + f"/{str}_Loss_After_Epoch_{epoch}_{currentTime}.png")
    plt.show()


def VisualizeAvgLoss(lossArr, folder, epoch, generator=True, SAVE=True, windowSize=5):
    x = (range(0, len(lossArr)))

    averageY = []
    sum = np.sum(lossArr[0:windowSize-1])
    for ind in range(len(lossArr) - windowSize + 1):
        sum += lossArr[ind+windowSize-1]
        averageY.append(sum/windowSize)
        sum -= lossArr[ind]

    for ind in range(windowSize - 1):
        averageY.insert(0, np.nan)

    plt.figure(figsize=(12, 10))
    plt.plot(x, averageY)
    str = "Discriminator"
    if generator:
        str = "Generator"

    plt.xlabel("Number of Iterations")
    plt.ylabel(str + " Loss")
    if SAVE:
        now = datetime.now()
        currentTime = now.strftime("%H:%M:%S")
        plt.savefig(
            folder + f"/{str}_Average_Loss_After_Epoch_{epoch}_WindowSize_{windowSize}_{currentTime}.png")
    plt.show()
