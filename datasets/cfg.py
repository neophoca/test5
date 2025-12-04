import os

root_a_xml = "./data/24_chromosomes_object/annotations"
root_a_img = "./data/24_chromosomes_object/images"

root_b = "./data/Autokary2022_1600x1600"

label_map = {
    "1": 1, "2": 2, "3": 3,
    "4": 4, "5": 5, "6": 6,
    "7": 7, "8": 8, "9": 9,
    "10": 10, "11": 11, "12": 12,
    "13": 13, "14": 14, "15": 15,
    "16": 16, "17": 17, "18": 18,
    "19": 19, "20": 20, "21": 21,
    "22": 22, "X": 23, "Y": 24,
    "A1": 1, "A2": 2, "A3": 3,
    "B4": 4, "B5": 5,
    "C6": 6, "C7": 7, "C8": 8, "C9": 9, "C10": 10, "C11": 11, "C12": 12,
    "D13": 13, "D14": 14, "D15": 15,
    "E16": 16, "E17": 17, "E18": 18,
    "F19": 19, "F20": 20,
    "G21": 21, "G22": 22,
}


max_size = 640
num_classes = 1 + 24  # background + 24 chromosomes