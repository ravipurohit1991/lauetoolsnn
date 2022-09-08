#!/bin/bash

export PATH=~/anaconda3/bin:$PATH
pkg='lauetoolsnn'
array=( 3.7 3.8 3.9 3.10 )


# delete old built packages
if [[ -d $HOME/conda-bld/ ]]; then
    rm -r $HOME/conda-bld/
fi
for i in $HOME/anaconda3/conda-bld/linux-64/$pkg*; do
    echo $i
    rm $i
done
echo "Deleting old conda packages done!"


# building conda packages
for i in "${array[@]}"
do
    echo $i
	conda build --py $i .
done
echo "Building conda packages done!"


# converting conda packages to other platforms
platforms=( osx-64 linux-32 linux-64 win-32 win-64 )
for file in $HOME/anaconda3/conda-bld/linux-64/$pkg*; do
    echo $file
    conda convert --platform all $file  -o $HOME/conda-bld/
    for platform in "${platforms[@]}"
    do
        conda convert --platform $platform $file  -o $HOME/conda-bld/
    done
done
echo "converting packages to other platforms done!"


# uploading packages
find $HOME/conda-bld/**/$pkg*.tar.bz2 | while read file
do
    anaconda upload $file
done
echo "Uploading conda packages done!"