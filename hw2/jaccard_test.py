import torch
from utils.box_utils import jaccard

def test(a, b):
    box_a = torch.tensor(a).unsqueeze(0)
    box_b = torch.tensor(b).unsqueeze(0)

    iou = jaccard(box_a, box_b).item()
    print(iou)


def main():
    test([0, 0, 12, 12], [0, 0, 12, 12])
    test([0, 0, 12, 12], [6, 0, 12, 12])
    test([6, 0, 12, 12], [0, 0, 12, 12])
    print()
    test([0, 0, 12, 12], [12, 12, 24, 24])
    test([0, 0, 12, 12], [13, 13, 24, 24])
    test([0, 0, 12, 12], [6, 6, 18, 18])



if __name__ == "__main__":
    main()