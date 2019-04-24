mkdir -p data/v1.0/

if [ "$1" = "faster_rcnn" ]; then
    wget https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_train.h5 -O ./data/v1.0/features_faster_rcnn_x101_train.h5
    wget https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_val.h5 -O ./data/v1.0/features_faster_rcnn_x101_val.h5
    wget https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_test.h5 -O ./data/v1.0/features_faster_rcnn_x101_test.h5
else
    echo "Usage: download_data_v1.sh faster_rcnn"
    exit 1
fi

download_and_extract() {
    URL=$1
    ZIP_FILE=$2
    wget $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d data/v1.0/
    rm $ZIP_FILE
}

wget https://s3.amazonaws.com/visual-dialog/data/v1.0/visdial_data_trainval.h5 -O ./data/v1.0/visdial_data_trainval.h5 
wget https://s3.amazonaws.com/visual-dialog/data/v1.0/visdial_params_trainval.json -O ./data/v1.0/visdial_params_trainval.json
wget https://www.dropbox.com/s/3knyk09ko4xekmc/visdial_1.0_val_dense_annotations.json?dl=0 -O ./data/v1.0/visdial_1.0_val_dense_annotations.json
download_and_extract https://www.dropbox.com/s/ix8keeudqrd8hn8/visdial_1.0_train.zip?dl=0 ./data/v1.0/visdial_1.0_train.zip
download_and_extract https://www.dropbox.com/s/ibs3a0zhw74zisc/visdial_1.0_val.zip?dl=0 ./data/v1.0/visdial_1.0_val.zip
download_and_extract https://www.dropbox.com/s/o7mucbre2zm7i5n/visdial_1.0_test.zip?dl=0 ./data/v1.0/visdial_1.0_test.zip

