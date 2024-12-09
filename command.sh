# Training command, data root for training dataset path
python train_lightning.py \
train_dataset.data_root=../LaRa/dataset/gobjaverse/gobjaverse.h5 \
test_dataset.data_root=../LaRa/dataset/gobjaverse/gobjaverse.h5 \
model.enable_residual_attribute=False

# residual version, data root for training dataset path
python train_lightning.py \
train_dataset.data_root=../LaRa/dataset/gobjaverse/gobjaverse.h5 \
test_dataset.data_root=../LaRa/dataset/gobjaverse/gobjaverse.h5 \
model.enable_residual_attribute=True

# Evaluation command
python eval_all.py