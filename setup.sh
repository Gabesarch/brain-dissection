#!/bin/sh
echo cloning repos...
if [ ! -d "dissect/" ] 
then
	git clone https://github.com/davidbau/dissect.git
else
	echo "Dissect repo already exists.. skipping clone.."
fi
if [ ! -d "XTConsistency/" ]
then
	git clone https://github.com/EPFL-VILAB/XTConsistency.git
else
	echo "XTC repo already exists... skipping..."
fi
echo downloading XTC checkpoints...
mkdir -p checkpoints
wget https://drive.switch.ch/index.php/s/QPvImzbbdjBKI5P/download && mv download checkpoints && unzip checkpoints/download && rm checkpoints/download
# wget https://drive.switch.ch/index.php/s/QPvImzbbdjBKI5P/download?path=%2F&files=rgb2normal_consistency.pth && mv rgb2normal_consistency.pth ./checkpoints
# wget https://drive.switch.ch/index.php/s/QPvImzbbdjBKI5P/download?path=%2F&files=rgb2reshading_consistency.pth && mv rgb2reshading_consistency.pth ./checkpoints
