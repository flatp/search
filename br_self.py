import pandas as pd
import csv
import numpy as np

with open("pop_direct.csv", 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    direct = list(reader)

with open("pop_data202312_m.csv", 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    stream = list(reader)

r_dash = [int(row[3]) for row in direct]
r = np.array(r_dash, dtype=float)
r = r / r.sum()

P_dash = []
for row1 in direct:
    tl1 = row1[1]
    line = []
    for row2 in direct:
        tl2 = row2[1]
        num = [int(row[3]) for row in stream if row[0] == tl1 and row[1] == tl2]
        if len(num) > 0:
            line.append(num[0])
        else:
            line.append(0)
    P_dash.append(line)

P = np.array(P_dash, dtype=float)
row_sums = P.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
P = P / row_sums

a = 0.2
I = np.eye(P.shape[0])
P = (1 - a) * P + a * I

b = 0
for i in range(2000):
    print(i)
    r_next = r @ P
    if np.linalg.norm((1 - b) * r_next + b * r - r) < 1e-6:
        break
    r = (1 - b) * r_next + b * r

r_list = r.tolist()
output =[]
for i in range(len(direct)):
    output.append([direct[i][1], r_list[i]])


with open('browserank_s_2_0.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(output)  # 横1行で保存