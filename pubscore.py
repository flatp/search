import csv

with open("publisher_burst_summary.csv", 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    publist = list(reader)

with open("news_all_score_true.csv", 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    newslist = list(reader)

with open("pop_all.csv", 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    namelist = list(reader)

with open("alllist.csv", 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    all = list(reader)

dic = {}
dic2 = {}
dic3 = {}

for row in newslist[1:]:
    name = row[0]
    name2 = [i for i in namelist if i[1] == name][0][0]
    pub = row[5]
    score = [i for i in publist if i[0] == pub][0]
    if name2 in dic:
        dic[name2] += float(score[7])/float(score[1])
    else:
        dic[name2] = float(score[7])/float(score[1])
    if name2 in dic2:
        dic2[name2] += float(score[6])/float(score[1])
    else:
        dic2[name2] = float(score[6])/float(score[1])
    if name2 in dic3:
        dic3[name2] += float(score[4])/float(score[1])
    else:
        dic3[name2] = float(score[4])/float(score[1])

output = [all[0]+["pub_score1", "pub_score2", "pub_score3"]]

for row in all[1:]:
    output.append(row+[dic[name], dic2[name], dic3[name]])
    

with open("alllist2.csv", 'w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(output)