# COMMANDS: the bash commands from some task
# TASK, REALIZATION, CONFIGURATION: variables passed by ducttape
submitter shell :: COMMANDS TASK_VARIABLES {
  action run {
    set +u  # needed to fix a virtualenv bug
    if [[ ! -z ${pyenv:-} ]]; then
      virtualenv=$pyenv

      # Load the environment
      if [[ $virtualenv == conda:* ]]; then
        target=$(echo $virtualenv | cut -d: -f2-)
        source ~/.bashrc
        conda deactivate
        conda activate $target
      else
        source $virtualenv
      fi
    fi
    set -u

    STARTED=$(date +%s)
    time eval "$COMMANDS"
    STOPPED=$(date +%s)
    TIME=$(($STOPPED - $STARTED))
    echo $TIME > ducttape_time.txt
    set -u
  }
}


submitter shell_gpu :: COMMANDS TASK_VARIABLES {
  action run {
    set +u  # needed to fix a virtualenv bug
    if [[ ! -z ${pyenv:-} ]]; then
      virtualenv=$pyenv

      # Load the environment
      if [[ $virtualenv == conda:* ]]; then
        target=$(echo $virtualenv | cut -d: -f2-)
        source ~/.bashrc
        conda deactivate
        conda activate $target
      else
        source $virtualenv
      fi
    fi
    set -u

    LOCK_DIR="$HOME/.lock"
	mkdir -p "$LOCK_DIR"
	if [ ! -f "$LOCK_DIR/master.lock" ]; then
		touch "$LOCK_DIR/master.lock"
	fi
	exec {MASTER}>"$LOCK_DIR/master.lock"
	flock -x ${MASTER}



	NUM_FOUND_GPUS=0

	while [[ $NUM_FOUND_GPUS < $devices_per_task ]]; do
		N_GPU=$(nvidia-smi -L | wc -l)
		FREE_GPUS=$(nvidia-smi | sed -e '1,/Processes/d' \
		  | tail -n+3 | head -n-1 | awk '{print $2}' \
		  | awk -v ng="$N_GPU" 'BEGIN{for (n=0;n<ng;++n){g[n] = 1}} {delete g[$1];} END{for (i in g) print i}')

		echo $FREE_GPUS
		NUM_FOUND_GPUS=0

		for DEVICE_ID in $(seq 0 $(($N_GPU - 1))); do
			echo "device $DEVICE_ID"
			if [[ $devices != *"${DEVICE_ID}"* ]]; then
				TMP=$(echo $FREE_GPUS | sed "s/$DEVICE_ID//")

				FREE_GPUS=$TMP
				echo "not in devices -> $FREE_GPUS"

				continue
			fi

			TMP_LOCK="$LOCK_DIR/$(cat /etc/hostname).$DEVICE_ID.lock"
			echo $TMP_LOCK
			if [ ! -f "$TMP_LOCK" ]; then
				touch "$TMP_LOCK"
			fi

			exec {CHECK}>$TMP_LOCK
			if ! flock -xn $CHECK ; then
				TMP=$(echo $FREE_GPUS | sed "s/$DEVICE_ID//")
				FREE_GPUS=$TMP
				echo "device $DEVICE_ID in use -> $FREE_GPUS"

			else
				NUM_FOUND_GPUS=$((NUM_FOUND_GPUS+1))
				echo "found gpus: $NUM_FOUND_GPUS"
			fi
			exec {CHECK}>&-

		done
		echo "end of loop"

		if [[ $NUM_FOUND_GPUS < $devices_per_task ]]; then
		echo "sleeping then restarting because found gpus ($NUM_FOUND_GPUS) < needed ($devices_per_task)"
		sleep 10s
		fi

	done

	CVD=" "
	iter=0
	for DEVICE in $FREE_GPUS; do
	  LOCK="$LOCK_DIR/$(cat /etc/hostname).$DEVICE.lock"
	  touch "$LOCK"
	  echo "locking $LOCK"
	  exec {CUR_FN}>$LOCK
	  flock -x $CUR_FN
	  CVD+="$DEVICE,$CVD"
	  iter=$((iter+1))
	  if [[ $iter == $devices_per_task ]]; then
	  echo "$iter == $devices_per_task"
	  break
	  fi
	done

	exec {MASTER}>&-
	export CUDA_VISIBLE_DEVICES=${CVD::-2}
	echo "cvd: $CUDA_VISIBLE_DEVICES"

    STARTED=$(date +%s)
    time eval "$COMMANDS"
    exec {CUR_FN}>&-
    STOPPED=$(date +%s)
    TIME=$(($STOPPED - $STARTED))
    echo $TIME > ducttape_time.txt
    set -u
  }
}

# COMMANDS: the bash commands from some task
# TASK, REALIZATION, CONFIGURATION: variables passed by ducttape
submitter sge :: action_flags
              :: COMMANDS
              :: TASK REALIZATION TASK_VARIABLES CONFIGURATION {
  action run {
    wrapper="ducttape_sge_job.sh"
    echo "#!/usr/bin/env bash" >> $wrapper
    echo "" >> $wrapper
    echo "#$ $resource_flags" >> $wrapper
    echo "#$ $action_flags" >> $wrapper
    echo "#$ -j y" >> $wrapper
    echo "#$ -o $PWD/job.out" >> $wrapper
    echo "#$ -N $CONFIGURATION-$TASK-$REALIZATION" >> $wrapper
    echo "" >> $wrapper

    # Bash flags aren't necessarily passed into the scheduler
    # so we must re-initialize them

    echo "set -euo pipefail" >> $wrapper
    echo "" >> $wrapper
    echo "$TASK_VARIABLES" | perl -pe 's/=/="/; s/$/"/' >> $wrapper

    # Setup the virtual environment
    cat >> $wrapper <<EOF

set +u  # needed to fix a virtualenv bug
if [[ ! -z \${pyenv:-} ]]; then
  virtualenv=\$pyenv

  # Load the environment
  if [[ \$virtualenv == conda:* ]]; then
    # I don't think we need this if we just do source activate
    # . /etc/profile.d/conda.sh
    target=\$(echo \$virtualenv | cut -d: -f2-)
    source deactivate
    #source activate \$target
    conda activate \$target
  else
    source \$virtualenv
  fi
fi
set -u

EOF
#	echo "location is $location"
	if [[ $location == "coe" ]]; then
		# Load the CUDA toolkit on COE grid
		echo "module load cuda10.1/toolkit/" >> $wrapper
	elif [[ $location == "clsp" ]]; then
		echo 'export CUDA_VISIBLE_DEVICES=`free-gpu`' >> $wrapper
	fi

    # The current working directory will also be changed by most schedulers
    echo "cd $PWD" >> $wrapper

    echo >> $wrapper
    echo "echo \"HOSTNAME: \$(hostname)\"" >> $wrapper
    echo "echo" >> $wrapper
    echo "echo CUDA in ENV:" >> $wrapper
    echo "env | grep CUDA" >> $wrapper
    echo "echo" >> $wrapper
#    echo "echo SGE in ENV:" >> $wrapper
#    echo "env | grep SGE" >> $wrapper
#    echo "nvidia-smi" >> $wrapper
    echo >> $wrapper

    echo "$COMMANDS" >> $wrapper
    echo "echo \$? > $PWD/exitcode" >> $wrapper  # saves the exit code of the inner process

	sleep $((1 + RANDOM % 11));

    # Use SGE's -sync option to prevent qsub from immediately returning
    qsub -V -S /bin/bash $wrapper | grep -Eo "Your job [0-9]+" | grep -Eo "[0-9]+" > $PWD/job_id
    job_id=`cat $PWD/job_id`

    # async job killer
    exitfn () {
      trap SIGINT
      echo "wait until I kill the job $job_id"
      qdel $job_id
      exit
    }

    trap "exitfn" INT

    # don't use -sync y, instead, wait on exitcode
    while [ ! -z "`qstat -u $USER | grep $job_id`" ]
    do
      sleep 15
    done

    trap SIGINT

    # restore the exit code saved from the inner process
    EXITCODE=$(cat $PWD/exitcode)
    [ $EXITCODE = "0" ]
  }
}
