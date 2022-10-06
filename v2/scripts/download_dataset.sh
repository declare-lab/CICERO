#!/bin/sh

# Create the directory
directory="../data"
if [ ! -d $directory ]; then
  mkdir $directory
else
  echo "Folder ${directory} already exists."
fi
cd $directory || exit
pwd

# Download CICERO v1
if [ ! -d "cicero_v1" ]; then mkdir "cicero_v1"; fi
for split in "train" "test" "val"; do
  wget "https://raw.githubusercontent.com/declare-lab/CICERO/main/data/${split}.json" --directory-prefix="cicero_v1" --timestamping
done

# Download CICERO v2
if [ ! -d "cicero_v2" ]; then mkdir "cicero_v2"; fi
for split in "train" "test" "val"; do
  echo "https://repo-url/${split}.json" --directory-prefix="cicero_v2" --timestamping
done
