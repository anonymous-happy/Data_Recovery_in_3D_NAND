# # Project name: Data Recovery in 3D NAND
# # Date modified: 02/05/2026
# #
# # This file analyzes the data leakage of a Tesla image as the read-offset operation shifts the offset voltage.
# # It plots data leakage versus offset voltage and shows the recovered image for each offset voltage value.
# #

import os

from PIL import Image, ImageOps
import re
from io import StringIO
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import csv
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns


pathfigure = os.getcwd() + '/Plot_results/'


def parse_data(filename):
    lines = []
    # This open, read and then close, so
    # it should not use too much memory for too long.
    with open(filename, 'r') as f:
        lines = f.readlines()

    f.close()

    # Extract the first column.
    # This is to prepare to use np.loadtxt.
    first_column = []
    dataa = []
    for line in lines:
        # This should not be longer than the first 100 characters, so
        # it should be pretty fast.
        first_index = line.find(':')
        last_index = line.find('P')
        first_column.append(line[first_index + 2:last_index - 1])
        comma_index = [i for i, d in enumerate(line) if d == ',']
        dataa.append(line[comma_index[2] + 1:])

    text = ''.join(dataa)
    s = StringIO(text)
    data = np.loadtxt(s, delimiter=',', dtype=np.int32)

    return first_column, data

def convert_dec_to_bin(berr):
    all_data = []
    for row in range(0, len(berr)):
        temp_all_data = []
        for col in range(0, len(berr[0])):
            temp_all_data.append(np.binary_repr(berr[row][col], width=8))
        all_data.append(''.join(temp_all_data))

    new_list = []
    for row in range(0, len(all_data)):
        new_list.append(','.join(all_data[row]))

    all_data = np.loadtxt(new_list, delimiter=',', dtype=np.int32)

    return all_data


def extract_file(file):
    set_feature, ber = parse_data(file)
    all_bit = convert_dec_to_bin(ber)
    return all_bit

def read_image(filename):
    lines = []
    # This open, read and then close, so
    # it should not use too much memory for too long.
    with open(filename, 'r') as f:
        lines = f.readlines()

    text = ''.join(lines)
    s = StringIO(text)
    data = np.loadtxt(s, delimiter=',', dtype=np.int32)

    return data

def show_image(data_l, select_, num, st):
    pixels_new = np.reshape(data_l, (133, 69))

    # # # Rotate
    pixels_rotate = np.rot90(pixels_new, 0)

    array = np.array(pixels_rotate, dtype=np.uint8)
    image = Image.fromarray(array)
    print('saving...')
    image.save(pathfigure + '{}.png'.format(st))

    return


# # ####################################################################################################################
# # ###################  Main File #####################################################################################
# # ####################################################################################################################
voltage_offset = np.arange(-1280, 1271, 10)

# Extract original Tesla image
image_tesla = read_image(os.getcwd() + '/Raw_data_files/3D_tesla_1page_size133x69_dec.txt')

# Extract read-out file
# Each row shows a targeted sanitized flash page after it has been read with an offset voltage value
# 256 rows correspond to 256 read-offset voltage values, ranging from −1280 mV to 1270 mV in increments of 10 mV
all_file = os.getcwd() + '/Raw_data_files/Read_Offset_blk28_page0_Recovery_bake_150C_120mins.txt'

# Parse data
set_feature, ber = parse_data(all_file)

# # re-arrage data file
# # set_features from 0 to 127 are Voffset from -1280mV to -10mV
# # set_features from 128 to 255 are Voffset from 0mV to 1270mV
adj_set_feature = np.hstack((set_feature[128:], set_feature[:128]))
adj_ber = np.vstack((ber[128:], ber[:128]))

# Calcualte raw bit error
total_bitflip = []
for j in range(0, len(adj_ber), 1):  # number of pages to read # row
    temp_bf = 0
    for column in range(0, len(adj_ber[0]), 1):
        if image_tesla[column] == adj_ber[j][column]:
            bitflip = 0
        else:  # no: count fbc
            difference = bin(adj_ber[j][column] ^ image_tesla[column])
            bitflip = int(difference[2:].count('1'))
        temp_bf += bitflip
    total_bitflip.append(temp_bf)

# Calculate data leakage %
new_bf = np.array(total_bitflip)/(len(ber[0]) * 8 * 0.01)


# # # Plot image of each offset voltages
for i in range(0, len(ber), 1):
    show_image(ber[i], 0, 1, 'Voff_{}V'.format(voltage_offset[i]))


# # # Plot offset voltage (V) vs. data leakage (%)
fig = plt.figure(1, figsize=(4, 2.5), dpi=800)  # plot size in inches (width, height) & resolution(DPI)

plt.plot(voltage_offset * 0.001, 100 - new_bf, 'o', markersize=1.5, color='blue')

plt.minorticks_on()
plt.grid(which='major', linestyle=':', linewidth=0.8)
plt.grid(which='minor', linestyle=':', linewidth=0.5)
plt.xlabel(r'$V_{offset}$ (V)', size=12)#, fontweight='bold')
plt.ylabel(r'$\eta_{leak}^{opt.}$ (%)', size=12)#, fontweight='bold')
# plt.xlim(0, 1.27)
# plt.ylim(45, 75)
plt.xticks(size=12)  # , fontweight='bold')
plt.yticks(size=12)#, fontweight='bold')
plt.tight_layout()
plt.savefig("Plot_data_leakage_vs_offset_voltage.png")
plt.close()
