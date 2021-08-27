#!/bin/bash

SCRIPTNAME="$(basename $0)"

OPTIND=1
while getopts "hd:c:s:" opt; do
  	case "$opt" in
	h)
		echo "$SCRIPTNAME: flags: -h help, -d data_root, -c ckpt_root, -s scene"
	        exit 0
		;;
	d)
	  	export DATA_ROOT="$OPTARG"
		;;
	c)
	  	export CKPT_ROOT="$OPTARG"
		;;
	s)
	  	export SCENE="$OPTARG"
		;;
	esac
done

export CONFIG_FILE=nerf_sh/config/tt
shift "$(( OPTIND -1 ))"

python -m octree.extraction \
  --train_dir $CKPT_ROOT/$SCENE/ --is_jaxnerf_ckpt \
  --config $CONFIG_FILE \
  --data_dir $DATA_ROOT/$SCENE/ \
  --output $CKPT_ROOT/$SCENE/octrees/tree.npz

python -m octree.optimization \
  --input $CKPT_ROOT/$SCENE/tree.npz \
  --config $CONFIG_FILE \
  --data_dir $DATA_ROOT/$SCENE/ \
  --output $CKPT_ROOT/$SCENE/octrees/tree_opt.npz
