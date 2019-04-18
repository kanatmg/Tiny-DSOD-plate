cd /home/itachi/arm/Tiny-DSOD
./build/tools/caffe train \
--solver="models/DCOD300/VOC0712/DCOD300_300x300/solver.prototxt" \
--gpu 0,1,2,3,4,5,6,7 2>&1 | tee jobs/DCOD300/VOC0712/DCOD300_300x300/DCOD300_VOC0712_DCOD300_300x300.log
