"""
=============================================
Title: sdss

Author(s): Alexandre Adam

Last modified: January 20, 2021

Description: Exploration of a few SDSS slices
=============================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
import os
import seaborn as sns

datapath = "../data/Skyserver_SQL1_20_2021 12 59 33 AM.csv"


def main():
    data = pd.read_csv(datapath, skiprows=1)
    # spec redshift is a better estimate (-> check this)
    # data["mean_redshift"] = data[["Spec_redshift", "Photo_redshift"]].mean(axis=1)

    # preprocessing
    data = data[data["Spec_redshift"] < 1.] # sharp drop of data after this (actually around 1.2)

    # spatial correlation should be measured in certain boxes of z, not the full box 0 < z < 1

if __name__ == "__main__":
    main()
