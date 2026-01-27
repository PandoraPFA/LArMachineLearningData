import argparse, math

from matplotlib import pyplot as plt

def main(args):
    train_x, train_y = [], []
    val_x, val_y = [], []
    with open(args.loss_file, "r") as f:
        for line in f:
            line = line.rstrip()
            train_or_val, x, y = line.split(" ")
            if train_or_val == "TRAIN":
                train_x.append(float(x))
                train_y.append(float(y))
            elif train_or_val == "VALID":
                val_x.append(float(x))
                val_y.append(float(y))
            else:
                raise ValueError(f"Line type {train_or_val} not recognised")

    if any(math.isnan(y_1) or math.isnan(y_2) for y_1, y_2 in zip(train_y, val_y)):
        print("Found nans, converting nan -> -0.1")
        train_y = [ y if not math.isnan(y) else -0.1 for y in train_y ]
        val_y = [ y if not math.isnan(y) else -0.1 for y in val_y ]

    _, ax = plt.subplots(1, 1, figsize=(8,6))
    ax.plot(train_x, train_y, c="k", alpha=0.6, label="Train")
    ax.plot(val_x, val_y, c="r", label="Val")
    ax.set_xlabel("Epoch", loc="right", fontsize=16)
    ax.set_ylabel("Loss", loc="top", fontsize=16)
    ax.grid()
    ax.legend(fontsize=14)
    plt.show()

def parse_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("loss_file", type=str)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main(parse_cli())
