import os

TRAIN = "train"
DEV = "dev"
# TEST = "test"
TEST = "dev"# For now, use dev set as test
LANGUAGES = ["english", "french", "irish", "italian", "spanish"]
RESOURCES = ["low"]#"medium"]
MODEL_TYPE = ["transformer"]#, "pointer_generator"]
EPOCHS_PER_RESOURCE = {"low": 1100, "medium": 400}
BATCH_SIZE_PER_RESOURCE = {"low": 64, "medium": 128}
EVAL_EVERY = 25
for model in MODEL_TYPE:
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
            checkpoints_folder = f"model-checkpoints-test/{model}/{language}/{resource}"
            pred_folder = f"predictions-test/{model}"
            logs_folder = "logs-test"

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
                  f"--train {data_folder}/{train_file} --dev {data_folder}/{valid_file} " +
                  f"--vocab {vocab_folder}/{vocab_file} --checkpoints-folder {checkpoints_folder}" +
                  f" >> {logs_folder}/train-log-{model}-{resource}-{language}.out")
            # Generate predictions for test set
            os.system(f"python generate.py --model-checkpoint {checkpoints_folder}/model_best.pth " +
                  f"--test {data_folder}/{covered_test_file} --vocab {vocab_folder}/{vocab_file} " +
                  f"--pred {pred_folder}/{pred_file}")
            # Evaluate accuracy of prediction file compared to true test set
            os.system(f"python evaluate.py --pred {pred_folder}/{pred_file} " +
                  f"--target {data_folder}/{test_file}")