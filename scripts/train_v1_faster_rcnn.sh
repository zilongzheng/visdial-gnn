python train.py --dataroot ./data/v1.0/ \
                --img_train features_faster_rcnn_x101_train.h5 \
                --img_val features_faster_rcnn_x101_val.h5 \
                --dialog_train visdial_1.0_train.json \
                --dialog_val visdial_1.0_val.json \
                --img_feat_size 2048