mkdir -p data/v0.9

if [ "$1" = "vgg16" ]; then
    wget https://s3.amazonaws.com/visual-dialog/data/v0.9/data_img_vgg16_pool5.h5 -O ./data/v0.9/data_img_vgg16_pool5.h5
elif [ "$1" = "vgg19" ]; then
    wget https://filebox.ece.vt.edu/~jiasenlu/codeRelease/visDial.pytorch/data/vdl_img_vgg.h5 -O ./data/v0.9/data_img_vgg19_pool5.h5
else
    echo "Usage: bash download_data_v09.sh [vgg16|vgg19]"
    exit 1
fi

wget https://s3.amazonaws.com/visual-dialog/data/v0.9/visdial_data.h5 -O ./data/v0.9/visdial_data.h5
wget https://s3.amazonaws.com/visual-dialog/data/v0.9/visdial_params.json -O ./data/v0.9/visdial_params.json


