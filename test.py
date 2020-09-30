from analyze_entropy import comp_entropy
import numpy as np

if __name__ == '__main__':
    x = np.asarray([0.89, 0.02] + [0.01] * 9)

    ent = comp_entropy(x, nucleus_filter=False)
    print(ent)

    ent = comp_entropy(x, nucleus_filter=True)

    v = [0.89 / 0.91, 0.02 / 0.91]
    out = sum([x * np.log(x) for x in v])
    print(out)
    print(ent)
