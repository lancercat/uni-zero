#DOWNLOAD=/run/media/lasercat/9f8924e7-76b4-42ab-bba0-6e31ee941a36/ococr-kaggle
DOWNLOAD=~/Downloads
mkdir ~/ssddata
mkdir ~/hydra_saves
mkdir ~/ssddata/anchors
unzip ${DOWNLOAD}/mose-extra -d ~/ssddata/
mkdir ~/ssddata/old
unzip ${DOWNLOAD}/osocrtraining -d ~/ssddata/old/
cd ~/ssddata/
mv Archive/* old/
mv athenaNG/ old/
mv dicts/ old/
cd old/
mv dicts/ ../
mv ssddata/* .
mv ssddata_1/* .
mv ssddata_2/* .
mv ../dicts/* dicts/
rmdir*
rm -rf NIPS2014 models-release CVPR2016 ctwch pami_ch_fsl_hwdb
# drop not used datasets.
rmdir *
mv dicts/dicts/* dicts/
cd ../
mv old/* .
rmdir *
cd ~/ssddata
cd abimjst/
mv abimjst/* ../
rmdir *
cd ../
rmdir *
cd ~/ssddata/dictsv2/
mv dictsv2/* .
rmdir *
cd ../
tar -xzf ${DOWNLOAD}/310-rel.tgz -C ~/hydra_saves/
cd ~/hydra_saves/
mv 310*/* .; rmdir *;
