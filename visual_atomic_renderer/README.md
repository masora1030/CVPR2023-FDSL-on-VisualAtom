# VisualAtom
Run the python script `render_VisualAtom.py`, you can get VisualAtom dataset.

## Requirements
- Python 3 (worked at 3.8.2)
- noise (worked at 1.2.2)
- PIL (worked at 9.0.0)

## Running the script to generate VisualAtom
Basically, you can run the script to generate VisualAtom with the following command.

```bash
$ python render_VisualAtom.py
```

The folder structure is constructed as follows.

```misc
./
  VisualAtom_dataset/
    image/
      00000/
        00000_0000.png
        00000_0001.png
        ...
      00001/
        00001_0000.png
        00001_0001.png
        ...
      ...
    ...
```

You can change the dataset folder name with --save_root. For a faster execution, you shuold run the bash as follows. You must adjust the thread parameter --numof_thread in the script depending on your computational resource.

Example script `make_VisualAtom.sh` (generating best practice VisualAtom-1k using 20 threads) is shown below.

```sh
SAVEDIR="./VisualAtom_dataset"
NUM_OF_THREAD=20

if [ ! -d ${SAVEDIR} ]; then
  mkdir ${SAVEDIR}
fi

# Multi-thread processing
for ((i=0 ; i<${NUM_OF_THREAD} ; i++))
do
  python render_VisualAtom \
  --save_root=${SAVEDIR} --numof_thread=${NUM_OF_THREAD} --thread_num=${i} \
  --numof_classes=1000 --numof_instances=1000 --start_class=0 \
  --vertex_num_min=200 --vertex_num_max=1000 --perlin_min=0 --line_num_min=1 --line_num_max=200 \
  --line_width=0.1 --radius_min=10 --oval_rate=2 --start_pos=512 \
  --nami_1_min=0 --nami_2_min=0 --nami_1_max=20 --nami_2_max=20 &

done
wait
```