import argparse
import data

# Arguments
import utils

parser = argparse.ArgumentParser(description='Computing accuracy of model predictions compare to target file')
parser.add_argument('--pred', type=str, default='data/pred',
                    help="File with model predictions (must include folder path)")
parser.add_argument('--target', type=str, default='target',
                    help="File with gold targets (must include folder path)")
args = parser.parse_args()

# Log all relevant files
logger = utils.get_logger()
logger.info(f"Target file: {args.target}")
logger.info(f"Prediction file: {args.pred}")


""" FUNCTIONS """
def accuracy(predictions, targets):
    """Return fraction of matches between two lists sequentially."""
    correct_count = 0
    for prediction, target in zip(predictions, targets):
        if prediction == target:
            correct_count += 1
    return float(100 * correct_count) / len(predictions)

def evaluate_predictions(pred_file, target_file):
    """Compute prediction. words NOT cleaned"""
    pred_lines = data.read_morph_file(pred_file)
    target_lines = data.read_morph_file(target_file)
    predictions = [line[1] for line in pred_lines]
    truth = [line[1] for line in target_lines]
    total_accuracy = accuracy(predictions, truth)
    logger.info(f"Test set. accuracy: {total_accuracy:.2f}\n")
    return total_accuracy


if __name__ == '__main__':
    # Compute accuracy of predictions compare to truth
    evaluate_predictions(args.pred, args.target)

