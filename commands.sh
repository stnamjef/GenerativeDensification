# training
python -m src.main +experiment=re10k data_loader.train.batch_size=2

# finetuning
python -m src.main +experiment=re10k \
data_loader.train.batch_size=2 \ 
checkpointing.load=checkpoints/fast_model.ckpt \
checkpointing.resume=false \
trainer.max_steps=150001

# --------------- Default Final Models ---------------

PRETRAINED_MODEL=checkpoints/final_model.ckpt
# PRETRAINED_MODEL=checkpoints/fast_model.ckpt

# RE10K evaluation
python -m src.main +experiment=re10k \
checkpointing.load=${PRETRAINED_MODEL} \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
dataset.roots=[./dataset/re10k]

# --------------- Cross-Dataset Generalization on ACID ---------------

# RealEstate10K -> ACID evaluation
python -m src.main +experiment=acid \
checkpointing.load=${PRETRAINED_MODEL} \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_acid.json \
test.compute_scores=true \
dataset.roots=[./dataset/acid]

# --------------- Cross-Dataset Generalization on DTU ---------------

# RealEstate10K -> DTU (2 context views) evaluation
python -m src.main +experiment=dtu \
checkpointing.load=${PRETRAINED_MODEL} \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_dtu_nctx2.json \
test.compute_scores=true \
dataset.roots=[./dataset/dtu]

# RealEstate10K -> DTU (3 context views) evaluation

python -m src.main +experiment=dtu \
checkpointing.load=${PRETRAINED_MODEL} \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_dtu_nctx3.json \
dataset.view_sampler.num_context_views=3 \
wandb.name=dtu/views3 \
test.compute_scores=true \
dataset.roots=[./dataset/dtu]