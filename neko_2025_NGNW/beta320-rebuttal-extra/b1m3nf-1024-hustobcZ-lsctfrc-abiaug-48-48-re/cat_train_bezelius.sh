#!/bin/bash
#SBATCH --gpus=1
#SBATCH -t 2-00:00:00

#Cmd to launch
#train.sh [gpuid]

IDRUN=$(uuidgen)
MTHDNAME=$(pwd |xargs basename );

echo "Create dir for log"
CURRENTDATE=`date +"%Y-%m-%d"`
echo "currentDate :"
echo $CURRENTDATE
echo "Running from commit:"
echo $(git rev-parse HEAD)

PATHLOGDIR="/home/x_liuch/logs/watch_and_build/${MTHDNAME}_allruns/${CURRENTDATE}_ID_${IDRUN}/"
MTHDLOGDIR="/home/x_liuch/logs/watch_and_build/${MTHDNAME}"
echo "path log dir:"
echo ${PATHLOGDIR}
mkdir -p ${PATHLOGDIR}
rm ${MTHDLOGDIR}
ln -sf ${PATHLOGDIR} ${MTHDLOGDIR}

PATHLOG=${PATHLOGDIR}"/PLAYDAN.log"
PATHMTHD=$(pwd)"/PLAYDAN.log"

export PYTHONPATH=$PYTHONPATH:../../../

PATHCONFIGGPU="/proj/document_analysis/users/shared/ostr_bezerlius_offlinewandb.json"

#">&1 means to redirect stderr(2) to stdout(1)
python3 train.py ${PATHCONFIGGPU} 2>&1 | tee ${PATHLOG}



