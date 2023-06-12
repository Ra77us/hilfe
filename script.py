import sys
import os
import jsonpickle
import numpy as np
from functools import lru_cache
import dtaidistance.dtw_ndim as dtw_2
import math
from more_itertools import locate

satart_idx = int(sys.argv[1])
print(np.__version__)
root_path = './' # you can change it to './' to make all files relative to current path, or use your drive path in Google colab
labels_file ='out_labels.json'
normalized_data_file = 'out_normalized_glyphs.json'


with open(labels_file, 'r') as file:
    labels_str = jsonpickle.decode(file.read())

label_str_dict = {}

def label_str_to_int(label):
    if label not in label_str_dict:
        i = len(label_str_dict)
        label_str_dict[label] = i
    return label_str_dict[label]

labels_int = [ label_str_to_int(label) for label in labels_str ]

class_names = [ list(label_str_dict.keys())[list(label_str_dict.values()).index(i)] for i in range(len(label_str_dict)) ]
class_amount = len(class_names)

inv_map = {v: k for k, v in label_str_dict.items()}

with open(normalized_data_file, 'r') as file:
    raw_glyphs = jsonpickle.decode(file.read())


# PROCESSING THE TIME SERIES
#

max_glyph_len = 0
for i, glyph in enumerate(raw_glyphs):
    glyph_len = 0
    for stroke in glyph:
        glyph_len += len(stroke)
    glyph_len //= 2
    if glyph_len > max_glyph_len:
        max_glyph_len = glyph_len


glyphs = []
for glyph in raw_glyphs:
    # combines strokes into one stroke, TODO?
    # combines pairs of coords x, y into tuples (x, y)
    # pads to equal length with (0, 0)
    new_glyph = []
    for stroke in glyph:
        for i in range(0, len(stroke), 2):
             new_glyph.append((stroke[i], stroke[i+1]))
    pad_len = max_glyph_len - len(new_glyph)
    # for _ in range(pad_len):
    #     new_glyph.append((0, 0)) # padding
    glyphs.append(new_glyph)

glyphs_np_arr = np.array(glyphs)


letters = [[] for _ in range(len(glyphs_np_arr))]
for i in range(len(glyphs)):
    point = []
    for j in range(len(glyphs_np_arr[i])):
        point.append(glyphs_np_arr[i][j][0])
        point.append(glyphs_np_arr[i][j][1])
    # print(point)
    letters[i] = point


def most_frequent(List):
    res = max(set(List), key = List.count)
    print(res)
    max_count = List.count(res)
    first_el = List[0]  
    if(List.count(first_el) == max_count):
        res = first_el
        print("zwracamy: ", res)
    return res

@lru_cache(maxsize=None)
def my_dist(x, y):
    p = dtw_2.distance(x, y)
    return p


class LabeledSample:
    def __init__(self, data, label):
        self.data = data
        self.label = label

class KNN:
    def __init__(self, samples, labels, k, dist_func):
        assert len(samples) == len(labels)
        self.data = [LabeledSample(samples[i], labels[i]) for i in range(len(samples))]
        self.k = k
        self.dist_func = dist_func
    
    def predict(self, sample):
        dists = []
        for j,d in enumerate(self.data):
            print(j)
            dists.append(self.dist_func(tuple(sample), tuple(d.data)))
        dists = np.array(dists)
        neighbors = [self.data[i] for i in np.argsort(dists)[:self.k]]
        # print(neighbors[0].data, neighbors[1].data)
        # są różne techniki siły głosu np głos ~ dystans, tutaj każdy sąsiad ma 1 głos
        labels = [n.label for n in neighbors]
        #print("Głosy: ", labels)
        return most_frequent(labels)
    
idxs_to_fit = []
idxs_to_test = []
# print(class_names)

for i in range(len(class_names)):
    array_with_idx = locate(labels_int, lambda x: x == i)
    l = list(array_with_idx)
    p = math.floor(0.8 * len(l))
    idxs_to_fit.extend(l[:p])
    idxs_to_test.extend(l[p:])

print(len(idxs_to_fit))
print(len(idxs_to_test))

idx = 0.8 * len(glyphs)
p = 9312
# print(idx)

set_to_fit = []
set_to_test = []

labels_to_fit = []
labels_to_test = []

for i in range(len(idxs_to_fit)):
    x = idxs_to_fit[i]
    # print(x)
    print(letters[x])
    set_to_fit.append(letters[idxs_to_fit[i]])
    labels_to_fit.append(labels_int[idxs_to_fit[i]])


for i in range(len(idxs_to_test)):
    set_to_test.append(letters[idxs_to_test[i]])
    labels_to_test.append(labels_int[idxs_to_fit[i]])


out = 'res.txt'

def run_test():
    for i in range(len(set_to_test)):
        print(f'{i}/{len(set_to_test)}')
        ans = []
        for k in [1, 5, 10, 15, 27, 49]:
            knn = KNN(np.array(set_to_fit), np.array(labels_to_fit), k, my_dist)
            idx = knn.predict(set_to_test[i])
            ans.append((idx, idx == labels_to_test[i]))
        line = f'{i} {inv_map[labels_to_test[i]]}:'
        for a in ans:
            line += (f'{a[0]},{a[1]};')
        with open(out, "a") as myfile:
            myfile.write(line + '\n')

run_test()