for runner in bin/runner*; do numactl --physcpubind=3 ./$runner 10000000 ; done
