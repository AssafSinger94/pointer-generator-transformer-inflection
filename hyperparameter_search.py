import os
import statistics

TRAIN = "train"
DEV = "dev"
# TEST = "test"
TEST = "dev"# For now, use dev set as test
LANGUAGES = ["english", "french", "irish", "italian", "spanish"]
RESOURCES = ["low"]#"medium"]
MODEL_TYPE = ["transformer"]#, "pointer_generator"]
EPOCHS_PER_RESOURCE = {"low": 800, "medium": 400}
BATCH_SIZE_PER_RESOURCE = {"low": 64, "medium": 128}
EVAL_EVERY = 25

EMBEDDING_DIMS = [64, 128, 256]
FCN_HIDDEN_DIMS = [64, 128, 256]
NUM_HEADS = [4, 8]
NUM_LAYERS = [2, 3]
# DROPOUT = [0.2, 0.3]
for model in MODEL_TYPE:
    for embed_dim in EMBEDDING_DIMS:
        for fcn_dim in FCN_HIDDEN_DIMS:
            for num_heads in NUM_HEADS:
                for num_layers in NUM_LAYERS:
                    # skip these
                    if (embed_dim, fcn_dim) == (64, 64):
                        continue
                    print(f"embed_dim: {embed_dim}, fcn_dim: {fcn_dim}, num-heads: {num_heads}, num-layers: {num_layers}")
                    accuracies = []
                    hyper_folder = f"embed_dim-{embed_dim}-fcn_dim-{fcn_dim}-heads-{num_heads}-layers-{num_layers}"
                    for resource in RESOURCES:
                        for language in LANGUAGES:
                            print(f"{resource} - {language}")
                            # Get epoch and batch size
                            epochs = EPOCHS_PER_RESOURCE[resource]
                            batch_size = BATCH_SIZE_PER_RESOURCE[resource]
                            # Set names of relevant files and directories
                            train_file = f"{language}-{TRAIN}-{resource}"
                            valid_file = f"{language}-{DEV}"
                            test_file = f"{language}-{TEST}"
                            covered_test_file = f"{language}-covered-{TEST}"
                            pred_file = f"{language}-{resource}-{TEST}-pred"
                            vocab_file = f"{train_file}-vocab"

                            data_folder = "data"
                            vocab_folder = f"vocab/{language}/{resource}"
                            checkpoints_folder = f"model-checkpoints-test/{model}/{hyper_folder}/{language}/{resource}"
                            pred_folder = f"predictions-test/{model}/{hyper_folder}"
                            logs_folder = f"logs-test/{hyper_folder}"

                            # create necessary folders, if they do not exist already
                            if not os.path.exists(vocab_folder):
                                os.makedirs(vocab_folder)
                            if not os.path.exists(checkpoints_folder):
                                os.makedirs(checkpoints_folder)
                            if not os.path.exists(pred_folder):
                                os.makedirs(pred_folder)
                            if not os.path.exists(logs_folder):
                                os.makedirs(logs_folder)

                            # Create vocabulary
                            # print(f"python vocabulary.py --src {data_folder}/{train_file} --vocab {data_folder}/{vocab_file}")
                            # Train model
                            os.system(f"python train.py --arch {model} --epochs {epochs} --batch-size {batch_size} --eval-every {EVAL_EVERY} " +
                                  f"--embed-dim {embed_dim} --fcn-dim {fcn_dim} --num-heads {num_heads} --num-layers {num_layers} " +
                                  f"--train {data_folder}/{train_file} --dev {data_folder}/{valid_file} " +
                                  f"--vocab {vocab_folder}/{vocab_file} --checkpoints-folder {checkpoints_folder}" +
                                  f" > {logs_folder}/train-log-{model}-{resource}-{language}.out")
                            # Generate predictions for test set
                            os.system(f"python generate.py --model-checkpoint {checkpoints_folder}/model_best.pth " +
                                  f"--test {data_folder}/{covered_test_file} --vocab {vocab_folder}/{vocab_file} " +
                                  f"--pred {pred_folder}/{pred_file}")
                            # Evaluate accuracy of prediction file compared to true test set
                            accuracies.append(os.system(f"python evaluate.py --pred {pred_folder}/{pred_file} " +
                                  f"--target {data_folder}/{test_file}"))
                    print(f"average accuracy: {statistics.mean(accuracies):.4f}")