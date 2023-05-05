SAVEDIR="./VisualAtom_dataset"
NUM_OF_THREAD=20

if [ ! -d ${SAVEDIR} ]; then
  mkdir ${SAVEDIR}
fi

# Multi-thread processing
for ((i=0 ; i<${NUM_OF_THREAD} ; i++))
do
  python render_VisualAtom.py \
  --save_root=${SAVEDIR} --numof_thread=${NUM_OF_THREAD} --thread_num=${i} \
  --numof_classes=1000 --numof_instances=1000 --start_class=0 \
  --vertex_num_min=200 --vertex_num_max=1000 --perlin_min=0 --line_num_min=1 --line_num_max=200 \
  --line_width=0.1 --radius_min=10 --oval_rate=2 --start_pos=512 \
  --nami_1_min=0 --nami_2_min=0 --nami_1_max=20 --nami_2_max=20 &

done
wait
