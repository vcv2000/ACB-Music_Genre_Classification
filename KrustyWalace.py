import scipy as sp
import numpy as np
import scipy.stats

def Krustywallece(B,C):

    if len(B) == 1:
        Means = B
    else:
        Means = []
        for i in range(len(B)):
            Means.append(B[i].mean())

    variables = C
    orderMeans = []
    ordervariables= []
    order = []
    for i in range(len(Means)):
        order.append({"Means":Means[i],"Variables":variables[i]})

    def sorting(a):
        return float(a["Means"])

    ordered = sorted(order,key=sorting)

    for i in ordered:
        orderMeans.append(i["Means"])
        ordervariables.append(i["Variables"])

    Dict = {}
    for i in range(len(ordervariables)):
        if ordervariables[i] in Dict.keys():
            Dict[ordervariables[i]].append(orderMeans[i])
        else:
            Dict[ordervariables[i]] = [orderMeans[i]]

    a = 0
    Ranking = []
    for i in range(len(orderMeans)):
        Ranking.append(i)

    i = 0
    c = 1
    ilis = []
    sum = 0
    while i < len(orderMeans):
        if i < len(orderMeans)-1:
            if orderMeans[i] == orderMeans[i+1]:
                ilis.append(i)
                sum += i+1
                c +=1
                i +=1
            else:
                if len(ilis) != 0:
                    for j in ilis:
                        Ranking[j] = (sum+1+i)/c
                    Ranking[i] = (sum+1+i)/c
                else:
                    Ranking[i] = i+1
                c = 1
                ilis = []
                sum = 0
                i+=1
        else:
            if len(ilis) != 0:
                for j in ilis:
                    Ranking[j] = (sum+1+i) / c
                Ranking[i] = (sum + 1 + i )/ c
            else:
                Ranking[i] = i + 1
            i += 1

    Rs = {}
    for i in range(len(Ranking)):
        if ordervariables[i] not in Rs.keys():
            Rs[ordervariables[i]] = [[Ranking[i]],1]
        else:
            Rs[ordervariables[i]][0].append(Ranking[i])
            Rs[ordervariables[i]][1] += 1

    Rsum = 0
    n = 0
    R = []
    for i in Rs.keys():
        Rum = 0
        for j in Rs[i][0]:
            n += 1
            Rsum += j
            Rum += j
        R.append([Rum/Rs[i][1],f"{i}",Rs[i][1]])

    Rmax = Rsum/n
    R.append([Rmax,"ALL",n])
    lastsum = 0
    for i in R:
        lastsum += i[2]*(i[0]-Rmax)**2
    H = (12/(R[-1][2]*(R[-1][2] +1)))*lastsum

    return R, H

