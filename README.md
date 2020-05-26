# Pointer-Generator-Transformer-Inflection-2019
Transformer model and pointer-generator transformer for the morphological inflection task

**MEDIUM RESOURCE TRAINING FILE - ENGLISH EXAMPLE**

*Data augmentation for dataset* - python augment.py --src "data/english-train-medium" --out "data/english-train-medium-aug"

*Create vocabulary for dataset* - python vocabulary.py --src "data/english-train-medium" --vocab "data/english-train-medium-vocab"

*Train model* - python train.py \
 --train "data/english-train-medium" --dev "data/english-dev" --vocab "data/english-train-medium-vocab" checkpoints-dir "checkpoints" \ 
 --batch-size 128 --epochs 200 --eval-every 1 --resume True \
 --arch transformer --embed-dim 64 --fcn-dim 256 --num-heads 4 --num-layers 2  --dropout 0.2 \ 
 --lr 0.001 --beta2 0.98 \
 --scheduler warmupinvsqr --patience 10 --min-lr 1e-5 --warmup-steps 4000

*Generate Predictions with model* - python generate.py \
--model-checkpoint "checkpoints/model_best.pth" \
--arch transformer --embed-dim 64 --fcn-dim 256 --num-heads 4 --num-layers 2  --dropout 0.2 \
--test "data/english-covered-test" \
--vocab "data/english-train-medium-vocab" \
--pred "data/english-test-pred-medium"

*Compute accuracy of test set predictions* - python evaluate.py \
--pred "data/english-test-pred-medium" --target "data/english-test"


**LOW RESOURCE TRAINING FILE - ENGLISH EXAMPLE**

*Data augmentation for dataset* - python augment.py --src "data/english-train-low" --out "data/english-train-low-aug"

*Create vocabulary for dataset* - python vocabulary.py --src "data/english-train-low" --vocab "data/english-train-low-vocab"

*Train model* - python train.py \
 --train "data/english-train-low" --dev "data/english-dev" --vocab "data/english-train-low-vocab" checkpoints-dir "checkpoints" \ 
 --batch-size 128 --epochs 200 --eval-every 1 --resume True \
 --arch transformer --embed-dim 64 --fcn-dim 256 --num-heads 4 --num-layers 2  --dropout 0.2 \ 
 --lr 0.001 --beta2 0.98 \
 --scheduler warmupinvsqr --patience 10 --min-lr 1e-5 --warmup-steps 4000

*Generate Predictions with model* - python generate.py \
--model-checkpoint "checkpoints/model_best.pth" \
--arch transformer --embed-dim 64 --fcn-dim 256 --num-heads 4 --num-layers 2  --dropout 0.2 \
--test "data/english-covered-test" \
--vocab "data/english-train-low-vocab" \
--pred "data/english-test-pred-low"

*Compute accuracy of test set predictions* - python evaluate.py \
--pred "data/english-test-pred-low" --target "data/english-test"
