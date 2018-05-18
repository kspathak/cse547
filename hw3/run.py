from cse547.data import CocoPatchesDataset

from absl import app

def main(argv):
    dataset = CocoPatchesDataset(
        '/data', 'train', 'tiny', supercategories = frozenset(['vehicle', 'animal']), iou_threshold=0.5)
    print(dataset.categories)
    print("hello, world!")

if __name__ == '__main__':
    app.run(main)
