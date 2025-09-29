# utils/bootstrap.py
# Common imports + device + seeding + light plotting defaults

# --- stdlib
import os, sys, random
from collections import Counter
from pathlib import Path

# --- core scientific
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.spatial.distance import euclidean
from scipy.stats import ttest_rel, wilcoxon

# --- neuro/plot
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns

# --- torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.manifold import TSNE

# --- globals
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42