# -- Twitter tokenization evaluation script
# -- Use this script to evaluate your tokenizer solution before submitting to the competition
# 
# -- Before running this script, make sure that:
# [1] You have installed the Levenshtein module: pip install levenshtein
# [2] You have correctly specified path to the ground truth and predicted data relative to this script
# [3] Finally, make sure that the format of the predicted file is correct - see the example submission file
# 
# -- Found bug?
# Let me (Ludek) know via Slack, thanks!

from Levenshtein import distance

PREDICT_PATH = "y_predicted_example.csv"
TRUE_PATH = "y_train.csv"

def evaluate():

    # -- Load the data
    with open(PREDICT_PATH, "r") as f:
        predict_lines = f.readlines()

    with open(TRUE_PATH, "r") as f:
        true_lines = f.readlines()

    # -- Compute the Levenstein distance between each line
    distances = []
    n = len(true_lines)
    for i in range(n):
        s1, s2 = true_lines[i], predict_lines[i]
        distances.append(distance(s1, s2))

    # -- Compute the Mean Levenstein distance and report it 
    result = sum(distances)/n
    print(f"Mean Levenstein distance: {result}")


if __name__ == "__main__":
    evaluate()