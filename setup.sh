#!/bin/sh
echo cloning repos...
git clone https://github.com/davidbau/dissect.git
git clone https://github.com/EPFL-VILAB/XTConsistency.git
echo downloading XTC checkpoints...
mkdir checkpoints
wget https://drive.switch.ch/index.php/s/QPvImzbbdjBKI5P/download?path=%2F&files=rgb2normal_consistency.pth
mv rgb2normal_consistency.pth ./checkpoints
wget https://drive.switch.ch/index.php/s/QPvImzbbdjBKI5P/download?path=%2F&files=rgb2reshading_consistency.pth
mv rgb2reshading_consistency.pth ./checkpoints