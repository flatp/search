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
r_d = np.array(r_dash, dtype=float)
r_d = r_d / r_d.sum()

P_dash = []
for row1 in direct:
    tl1 = row1[1]
    line = []
    for row2 in direct:
        tl2 = row2[1]
        num = [1/int(row[3]) for row in stream if row[0] == tl1 and row[1] == tl2]
        if len(num) > 0:
            line.append(num[0])
        else:
            line.append(0)
    P_dash.append(line)

P = np.array(P_dash, dtype=float)
row_sums = P.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
P = P / row_sums
print(P.shape)

row_sums_after_norm = P.sum(axis=1)  # 正規化後の各行の合計（通常は1か0のはず）
additional_column = np.where(row_sums_after_norm > 0, 0, 1)
P = np.hstack((P, additional_column.reshape(-1, 1)))
print(P.shape)

r = np.append(r_d, 0.0)
P = np.vstack((P, r.reshape(1, -1)))
print(P.shape)

pi = np.ones_like(r) / r.size

alpha = 0.85
for i in range(2000):
    pi_next = alpha * pi @ P + (1 - alpha) * r
    if np.linalg.norm(pi_next - pi, ord=1) < 1e-8:
        print(i)
        pi = pi_next
        break
    pi = pi_next

pi = pi[:-1] 
pi = pi / np.abs(r_d)
pi = pi / pi.sum()
r_list = pi.tolist()
output =[]
for i in range(len(direct)):
    output.append([direct[i][1], r_list[i]])


with open('browserank_true_r.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(output)  # 横1行で保存