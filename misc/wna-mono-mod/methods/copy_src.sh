for i in $(sh find_srcdirs.sh);
do
     R=$(basename $(dirname ${i}));
     M=$(basename ${i});
     rm -rf ${R}/${M};
     mkdir -p ${R}/${M};
     cp ${i}/train.py ${R}/${M}/
     cp ${i}/test.py ${R}/${M}/
     cp ${i}/loadout.py ${R}/${M}/
done
