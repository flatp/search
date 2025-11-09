import csv

with open("weighted_pageviews.csv", 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    l1 = list(reader)

with open("weighted_pageviews2.csv", 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    l2 = list(reader)

with open("weighted_pageviews3.csv", 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    l3 = list(reader)

with open("weighted_pageviews4.csv", 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    l4 = list(reader)

with open("weighted_pageviews5.csv", 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    l5 = list(reader)

with open("alllist2.csv", 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    all = list(reader)

output = [all[0]+["weightedPV1", "weightedPV2", "weightedPV3", "weightedPV4", "weightedPV5"]]

for row in all[1:]:
    sc1 = [i[1] for i in l1 if row[0] == i[0]][0]
    sc2 = [i[1] for i in l2 if row[0] == i[0]][0]
    sc3 = [i[1] for i in l3 if row[0] == i[0]][0]
    sc4 = [i[1] for i in l4 if row[0] == i[0]][0]
    sc5 = [i[1] for i in l5 if row[0] == i[0]][0]
    output.append(row+[sc1, sc2, sc3, sc4, sc5])
    

with open("alllist3.csv", 'w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(output)