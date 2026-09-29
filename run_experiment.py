"""One-click VS Code run of the entire small pilot."""
from prepare_math_data import run as prepare
from train import run as train
from evaluate import run as evaluate
from compare import run as compare
from screen_report import run as screen_report


if __name__ == "__main__":
    prepare()
    train()
    evaluate()
    compare()
    screen_report()
