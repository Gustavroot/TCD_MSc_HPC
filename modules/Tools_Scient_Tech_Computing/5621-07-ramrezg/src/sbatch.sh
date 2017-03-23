#!/bin/bash

#SBATCH -n 8        # 8 cores = 1 node on lonsdale
#SBATCH -p compute
#SBATCH -t 16:00:00  # 16 hours
#SBATCH -U mschpc
#SBATCH -J all-combinations-8-cores
#SBATCH --reservation=application

# source the module commands
source /etc/profile.d/modules.sh

#script running all possible configurations for 8-cores
./general.sh 8
