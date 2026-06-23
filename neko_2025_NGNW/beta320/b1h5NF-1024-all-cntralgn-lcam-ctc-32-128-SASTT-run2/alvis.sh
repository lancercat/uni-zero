#!/bin/bash
#SBATCH -t 2-00:00:00
# SBATCH -A  NAISS2025-5-540

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

PATHLOGDIR="/mimer/NOBACKUP/groups/document_analysis_alvis/chang_assets/hydra_logs_extra/watch_and_build/${MTHDNAME}_allruns/${CURRENTDATE}_ID_${IDRUN}/"
MTHDLOGDIR="/mimer/NOBACKUP/groups/document_analysis_alvis/chang_assets/hydra_logs_extra/watch_and_build/${MTHDNAME}"
echo "path log dir:"
echo ${PATHLOGDIR}
mkdir -p ${PATHLOGDIR}
rm ${MTHDLOGDIR}
ln -sf ${PATHLOGDIR} ${MTHDLOGDIR}

PATHLOG=${PATHLOGDIR}"/PLAYDAN.log"
PATHMTHD=$(pwd)"/PLAYDAN.log"

export PYTHONPATH=$PYTHONPATH:../../../
source ~/venvs/osocr/bin/activate

PATHCONFIGGPU="/mimer/NOBACKUP/groups/document_analysis_alvis/chang_assets/alvispath.json"

#">&1 means to redirect stderr(2) to stdout(1)
python3 train.py ${PATHCONFIGGPU} 2>&1 | tee ${PATHLOG}



