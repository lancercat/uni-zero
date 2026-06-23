for i in $(cat method.txt | grep "_"); do find /home/lasercat/cat/neko_wcki/neko_2024_NGNW/*/${i} -type d | grep -v cache | grep -v e2$; done
