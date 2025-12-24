import colorsys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"

root_a_xml = DATA_ROOT / "24_chromosomes_object" / "annotations"
root_a_img = DATA_ROOT / "24_chromosomes_object" / "images"
root_d_xml = DATA_ROOT / "single_chromosomes_object" / "annotations"
root_d_img = DATA_ROOT / "single_chromosomes_object" / "images"

# shared metadata (provided once)
train_txt = DATA_ROOT / "train.txt"
test_txt = DATA_ROOT / "test.txt"
diff_txt = DATA_ROOT / "diff_image.txt"
normal_csv = DATA_ROOT / "normal.csv"
number_csv = DATA_ROOT / "number_abnormalities.csv"
structural_csv = DATA_ROOT / "structural_abnormalities.csv"

root_b = DATA_ROOT / "Autokary2022_1600x1600"
root_c = DATA_ROOT / "Chromo-CRCN"

root_custom = DATA_ROOT / "Custom"

label_map = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "10": 10, "11": 11, "12": 12, "13": 13, "14": 14, "15": 15, "16": 16, "17": 17, "18": 18,
    "19": 19, "20": 20, "21": 21, "22": 22, "X": 23, "Y": 24,
    "A1": 1, "A2": 2, "A3": 3, "B4": 4, "B5": 5,
    "C6": 6, "C7": 7, "C8": 8, "C9": 9, "C10": 10, "C11": 11, "C12": 12,
    "D13": 13, "D14": 14, "D15": 15, "E16": 16, "E17": 17, "E18": 18,
    "F19": 19, "F20": 20, "G21": 21, "G22": 22,
}

class_colors = {}
num_pairs = 11
for pair in range(num_pairs):
    h = pair / num_pairs
    for j in range(2):
        cid = 2 * pair + j + 1
        s = 0.9
        v = 1.0 if j == 0 else 0.7
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        class_colors[cid] = (r, g, b)

class_colors[23] = (1.0, 0.0, 1.0)
class_colors[24] = (0.0, 1.0, 1.0)

max_size = 640
num_classes = 1 + 24
