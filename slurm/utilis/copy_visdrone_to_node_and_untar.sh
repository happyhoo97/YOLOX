#!/user/bin/bash

for dataset in VisDrone; do
    p_local="/local_datasets/${dataset}"
    if [! -d "{p_local}"]; then
        tarfile = "/data/datasets/${dataset}.tar"
        if [! -f "$tarfile"]; then
            echo "$tarfile does not exsit."
            exit 1
        fi

        echo "mkdir ${p_local} ..."
        mkdir $p_local

        echo "cd to ${p_local} ..."
        cp $tarfile .

        echo "untar ..."
        tar -xf "${dataset}.tar"

        echo -e "done.\n\n"
    fi
done