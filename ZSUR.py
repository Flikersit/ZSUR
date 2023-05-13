import math
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from numpy. linalg import norm

def nacteny(body):
    xarray = []
    yarray = []
    for i in range(len(body)):
        xarray.append(body[i][0])
        yarray.append(body[i][1])
    return [xarray, yarray]


def otevrit(cesta):# metoda ktera otevre soubor
    data = []
    file = open(cesta, 'r')
    for line in file:
        numbers = line.split()
        number1 = numbers[0].split('e')
        number2 = numbers[1].split('e')
        if number1[1][1] == '0':
            final_num_1 = float(number1[0]) * 10**(float(number1[1][0] + number1[1][2]))
        else:
            final_num_1 = float(number1[0] * 10**(float(number1[1])))
        if number2[1][1] == '0':
            final_num_2 = float(number2[0]) * 10 ** (float(number2[1][0] + number2[1][2]))
        else:
            final_num_2 = float(number2[0] * 10 ** (float(number2[1])))
        data.append(np.array([final_num_1, final_num_2]))
    file.close()
    return data


def mintr(tr1, tr2):
    vzdalenost = []
    for i in range(len(tr1)):
        for j in range(len(tr2)):
            vzdalenost.append(norm(tr1[i] - tr2[j]))
    return min(vzdalenost)


def shlukhlad(data):# metoda shlukove hladiny
    vzdalenost = []
    shluk = []
    for i in range(len(data)):
        shluk.append([data[i]])
    while len(shluk) != 1:
        mezivzd = []
        rand = rd.randint(0, len(shluk) - 1)
        for i in range(len(shluk)):
            if i != rand:
                mezivzd.append(mintr(shluk[rand], shluk[i]))
        mens = min(mezivzd)
        vzdalenost.append(mens)
        inx = mezivzd.index(mens)
        if inx >= rand:
            inx += 1
        newtri = []
        for i in range(len(shluk[rand])):
            newtri.append(shluk[rand][i])
        for i in range(len(shluk[inx])):
            newtri.append(shluk[inx][i])
        shluk.pop(min(inx, rand))
        shluk.pop(max(inx, rand) - 1)
        shluk.append(newtri)
    m = max(vzdalenost)
    pocet = 0
    for i in range(len(vzdalenost)):
        if vzdalenost[i] >= 0.7*m:
            pocet += 1
    print("Pocet trid", pocet + 1)
    return vzdalenost


def mapa(data, q):#q=175 pro moje data, metoda retezove mapy
    tridy = []
    vzdalenost = []
    prvky = []
    aktualni = rd.randint(0, len(data))
    prvky.append(aktualni)
    vzdalenost.append(0)
    while len(prvky) != len(data):
        vztahy = []
        elementy = []
        for g in range(len(data)):
            if g not in prvky:
                v1 = data[aktualni] - data[g]
                vztahy.append(np.dot(v1, v1))
                elementy.append(g)
        vzdalenost.append(min(vztahy))
        prvky.append(elementy[vztahy.index(min(vztahy))])
        aktualni = elementy[vztahy.index(min(vztahy))]
    stred = sum(vzdalenost)*q/(len(vzdalenost))
    pocet_trid = 0
    tridy.append([])
    for g in range(len(vzdalenost)):
        if vzdalenost[g] > stred:
            pocet_trid += 1
            tridy.append([])
        tridy[pocet_trid].append(data[prvky[g]])

    print('Pocet trid', pocet_trid + 1)
    return tridy


def maxmin(data, q):# metoda maxmin, vrati seznam seznamu trid
    tridy = []
    stredy = []
    prvni = rd.randint(0, len(data))
    stredy.append(data[prvni])
    vzdalenost = []
    for i in range(len(data)):
        v = data[prvni] - data[i]
        vzdalenost.append(norm(v))
    stredy.append(data[vzdalenost.index(max(vzdalenost))])
    tridy.append([data[prvni]])
    tridy.append([data[vzdalenost.index(max(vzdalenost))]])
    while True:
        for i in range(len(stredy)):
            tridy[i] = []
        vzdalenost = []
        vzdalenost0 = []
        for i in range(len(data)):
            vzdalenost1 = []
            for k in range(len(stredy)):
                v = stredy[k] - data[i]
                vzdalenost1.append(norm(v))
            vzdalenost0.append(vzdalenost1)
        for i in range(len(vzdalenost0)):
            prvek = min(vzdalenost0[i])
            tridy[vzdalenost0[i].index(prvek)].append(data[i])
        for i in range(len(stredy)):
            stredy[i] = sum(tridy[i])/len(tridy[i])
        for k in range(len(tridy)):
            for g in range(len(tridy[k])):
                v = stredy[k] - tridy[k][g]
                vzdalenost.append(np.dot(v, v.transpose()))
        maximum = max(vzdalenost)
        flag = True
        vzdalenoststredu = []
        for i in range(len(stredy)):
            for k in range(i+1, len(stredy)):
                delka = stredy[i] - stredy[k]
                vysl = np.dot(delka, delka)
                vzdalenoststredu.append(vysl)
        if q*max(vzdalenoststredu) > maximum:
            flag = False
        if not flag:
            print("Pocet trid:", len(stredy))
            return tridy
        klas = 0
        pozice = 0
        for k in range(len(tridy)):
            for g in range(len(tridy[k])):
                v = stredy[k] - tridy[k][g]
                if np.dot(v, v) == maximum:
                    klas = k
                    pozice = g
        stredy.append(tridy[klas][pozice])
        tridy.append([tridy[klas][pozice]])
        tridy[klas].pop(pozice)
        for i in range(len(stredy)):
            stredy[i] = sum(tridy[i])/len(tridy[i])


def kmeans(data, clasy):# metoda kmeans, vstup: data, tridy , vystup: seznam seznamu trid
    tridy = []
    vzd = []
    stredy = []
    J = 0
    inde = []
    Jpred = 0
    for i in range(clasy):
        tridy.append([])
    for i in range(clasy):
        nahoda = rd.randint(0, len(data))
        while nahoda in inde:
            nahoda = rd.randint(0, len(data))
        inde.append(nahoda)
        stredy.append(data[nahoda])
    for i in range(len(data)):
        sezvz = []
        for k in range(len(stredy)):
            v = data[i] - stredy[k]
            sezvz.append(norm(v)**2)
        vzd.append(sezvz)
    for i in range(len(data)):
        mi = min(vzd[i])
        J += mi
        ind = vzd[i].index(mi)
        tridy[ind].append(data[i])
    for i in range(len(stredy)):
        stredy[i] = sum(tridy[i])/len(tridy[i])
    delJ = J
    while delJ != 0:
        for i in range(len(tridy)):
            tridy[i] = []
        vzd.clear()
        Jpred = J
        J = 0
        for i in range(len(data)):
            sezvz = []
            for k in range(len(stredy)):
                v = data[i] - stredy[k]
                sezvz.append(norm(v)**2)
            vzd.append(sezvz)
        for i in range(len(data)):
            mi = min(vzd[i])
            J += mi
            ind = vzd[i].index(mi)
            tridy[ind].append(data[i])
        for i in range(len(stredy)):
            stredy[i] = sum(tridy[i]) / len(tridy[i])
        delJ = Jpred - J
    return [tridy, J]


def kmeannebin(data):# binarni deleni kmeans, vstup: data, vystup: seznam seznamu (Funguje pouze pro 3 tridy!!!)
    tri12 = kmeans(data, 2)[0]
    stredy = []
    vzdalenost = [[], []]
    for i in range(len(tri12)):
        stredy.append(sum(tri12[i])/len(tri12[i]))
    for i in range(len(tri12)):
        for j in range(len(tri12[i])):
            vzdalenost[i].append(norm(stredy[i] - tri12[i][j]))
    summa1 = sum(vzdalenost[0])
    summa2 = sum(vzdalenost[1])
    if summa1 > summa2:
        ind = 0
        indr = 1
    else:
        ind = 1
        indr = 0
    tri22 = kmeans(tri12[ind], 2)[0]
    vys = [tri12[indr], tri22[0], tri22[1]]
    stredy_vys = []
    for k in range(len(vys)):
        stredy_vys.append(sum(vys[k]) / len(vys[k]))
    J = 0
    for l in range(len(vys)):
        for d in range(len(vys[l])):
            v = vys[l][d] - stredy_vys[l]
            J += norm(v) ** 2
    print("Kriterium pri binarnim deleni", J)
    return vys


def find(Tr, x): #hleda prvek v seznamu seznamu, vstup:seznam seznamu, hledani prvek, vystup: indexy
    for u in range(len(Tr)):
        T = Tr[u]
        for i in range(len(T)):
            if (T[i].all == x.all):
                return u,i


def krit(h, vtr, i):# podminka presunuti, vstup: seznam vzdalenosti, seznam sezznamu, index tridy do ktere patri prvek
    A = []          # vystup: index tridy do ktrere pridame prvek
    E = []
    for k in range(len(vtr)):
        if k == i:
            A.append(len(vtr[k])*h[k]/(len(vtr[k]) - 1))
        else:
            A.append(len(vtr[k]) * h[k] / (len(vtr[k]) + 1))
            E.append(len(vtr[k]) * h[k] / (len(vtr[k]) + 1))
    el = min(E)
    if el < A[i]:
        return [i, A.index(el)]
    else:
        return [-1, -1]


def optim(tridy):# metoda iterativni optimalizace, vstup: seznam seznamu, vystup: seznam seznamu
    stredy = []
    stredy_poc = []
    vtr = tridy.copy()
    for m in range(len(tridy)):
        stredy.append(sum(vtr[m])/len(vtr[m]))
        stredy_poc.append(sum(vtr[m])/len(vtr[m]))
    for i in range(len(tridy)):
        for j in range(len(tridy[i])):
            h = []
            for k in range(len(stredy)):
                v = tridy[i][j] - stredy[k]
                h.append(norm(v)**2)
            vys = krit(h, vtr, i)
            if vys[0] != -1:
                ind = find(vtr, tridy[i][j])
                vtr[vys[1]].append(vtr[ind[0]][ind[1]])
                del vtr[ind[0]][ind[1]]
                for n in range(len(tridy)):
                    stredy[n] = (sum(vtr[n]) / len(vtr[n]))
    J = 0
    J_po = 0
    for l in range(len(tridy)):
        for d in range(len(tridy[l])):
            v = tridy[l][d] - stredy_poc[l]
            J += norm(v)**2
    for u in range(len(vtr)):
        for p in range(len(vtr[u])):
            v = vtr[u][p] - stredy[u]
            J_po += norm(v)**2
    print("Celkova hodnota kriteria do optimalizace", J)
    print("Celkova hodnota kriteria po optimalizace", J_po)
    return vtr


def pravbayes(tridy, x):#Bayessuv klas, vstup: seznam seznamu, prvek, vystup: vektor s nulami a jednou jednickou
    stredy = []
    c = []
    pocet = 0
    pravpod = []#apriorni pravdepodobnost
    vysl = []
    elements = []
    for i in range(len(tridy)):
        prvky = []
        for j in range(len(tridy[i])):
            pocet += 1
            prvky.append(tridy[i][j])
        stredy.append(sum(prvky)/len(prvky))
    for i in range(len(tridy)):
        pravpod.append(len(tridy[i]) / pocet)#apriorni pravdepodobnost
        prvky = []
        for j in range(len(tridy[i])):
            v = tridy[i][j] - stredy[i]
            prvky.append(np.dot(v, v.T))
        c.append(np.diag((1, 1))*sum(prvky)/len(tridy[i]))
    for i in range(len(tridy)):
        v = x - stredy[i]
        soucin = np.dot(v.T, np.linalg.inv(c[i]))
        res = 1/math.sqrt((2*math.pi) ** 2) * 1/math.sqrt(np.linalg.det(abs(c[i]))) * math.exp((-1) * 1/2 * np.dot(soucin, v))
        elements.append(res*pravpod[i])#apriorni pravdepodobnost
    ind = elements.index(max(elements))
    for i in range(len(tridy)):
        if i != ind:
            vysl.append(0)
        else:
            vysl.append(1)
    return vysl


def jedns(tridy, x):# jeden nejblissi soused, vstup:seznam seznamu, prvek, vystup: vektor s nulami a jednou jednickou
    vys = []
    delka = []
    vzdal_tr = []
    for i in range(len(tridy)):
        prvky = []
        for j in range(len(tridy[i])):
            v = x - tridy[i][j]
            test = norm(x - tridy[i][j])
            prvky.append(test)
            delka.append(test)
        vzdal_tr.append(prvky)
    hodnota = min(delka)
    for i in range(len(tridy)):
        if hodnota in vzdal_tr[i]:
            vys.append(1)
        else:
            vys.append(0)
    return vys


def dvs(tridy, x):# podle dvou nejblissich sousedu vstup:seznam seznamu, prvek, vystup: vektor s nulami a jednou jednickou
    vzdal_tr = []
    vys = []
    for i in range(len(tridy)):
        vzd = []
        for k in range(len(tridy[i])):
            v = tridy[i][k] - x
            vzd.append(norm(v))
        min1 = min(vzd)
        vzd.remove(min1)
        min2 = min(vzd)
        vzdal_tr.append((min1 + min2)/2)
    ind = vzdal_tr.index(min(vzdal_tr))
    for i in range(len(tridy)):
        if i == ind:
            vys.append(1)
        else:
            vys.append(0)
    return vys


def kvant(tridy, x):#vektorova kvantizace vstup:seznam seznamu, prvek, vystup: vektor s nulami a jednou jednickou
    vys = []
    stredy = []
    for i in range(len(tridy)):
        prvky = []
        for j in range(len(tridy[i])):
            prvky.append(tridy[i][j])
        stredy.append(sum(prvky)/len(prvky))
    vzd = []
    for i in range(len(stredy)):
        v = x - stredy[i]
        vzd.append(math.sqrt(np.dot(v, v.T)))
    mi = vzd.index(min(vzd))
    for i in range(len(stredy)):
        if i == mi:
            vys.append(1)
        else:
            vys.append(0)
    return vys, stredy


def trenros(tridy):#rosenblatuv algoritmus  vstup: seznam seznamu, vystup: pocet iteraci, parametry klasifikatoru
    iterace = 0
    q = []
    for i in range(3):
        nah1 = rd.randint(-1, 1)
        nah2 = rd.randint(-1, 1)
        nah3 = rd.randint(-1, 1)
        q.append(np.array([nah1, nah2, nah3]))
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    while True:
        for i in range(len(tridy)):
            for j in range(len(tridy[i])):
                iterace += 1
                xv = np.array([1, tridy[i][j][0], tridy[i][j][1]])
                t1 = 0
                t2 = 0
                t3 = 0
                if i == 0:
                    t1 = 1
                    t2 = -1
                    t3 = -1
                elif i == 1:
                    t1 = -1
                    t2 = 1
                    t3 = -1
                else:
                    t1 = -1
                    t2 = -1
                    t3 = 1
                vyr1 = np.dot(q1.T, xv) * t1
                if 0 > vyr1:
                    q1 = q1 + t1*xv
                vyr2 = np.dot(q2.T, xv) * t2
                if 0 > vyr2:
                    q2 = q2 + t2*xv
                vyr3 = np.dot(q3.T, xv) * t3
                if 0 > vyr3:
                    q3 = q3 + t3 * xv
        chyba = []
        for i in range(len(tridy)):
            for j in range(len(tridy[i])):
                xv = np.array([1, tridy[i][j][0], tridy[i][j][1]])
                t1 = 0
                t2 = 0
                t3 = 0
                if i == 0:
                    t1 = 1
                    t2 = -1
                    t3 = -1
                elif i == 1:
                    t1 = -1
                    t2 = 1
                    t3 = -1
                else:
                    t1 = -1
                    t2 = -1
                    t3 = 1
                vyr1 = np.dot(q1.T, xv) * t1
                if 0 > vyr1:
                    chyba.append(1)
                vyr2 = np.dot(q2.T, xv) * t2
                if 0 > vyr2:
                    chyba.append(1)
                vyr3 = np.dot(q3.T, xv) * t3
                if 0 > vyr3:
                    chyba.append(1)
        if 1 not in chyba:
            break
    return [q1, q2, q3, iterace]


def testumkp(c, x, q, tr):# metoda urcuje nutny pocet iteraci pro spravnou klasofokaci
    count = 0
    while True:
        count += 1
        qk = q + c*x*tr*count
        vyr = np.dot(qk.T, x)*tr
        if vyr > 0:
            break
    return count


def umkp(tridy, c):# upravena metoda konstantnich prirustku vstup: seznam seznamu, konstanta uceni vystup: nastaveni klasifikatoru
    iterace = 0
    q = []
    for i in range(3):
        nah1 = rd.randint(-1, 1)
        nah2 = rd.randint(-1, 1)
        nah3 = rd.randint(-1, 1)
        q.append(np.array([nah1, nah2, nah3]))
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    while True:
        for i in range(len(tridy)):
            for j in range(len(tridy[i])):
                iterace += 1
                xv = np.array([1, tridy[i][j][0], tridy[i][j][1]])
                t1 = 0
                t2 = 0
                t3 = 0
                if i == 0:
                    t1 = 1
                    t2 = -1
                    t3 = -1
                elif i == 1:
                    t1 = -1
                    t2 = 1
                    t3 = -1
                else:
                    t1 = -1
                    t2 = -1
                    t3 = 1
                vyr1 = np.dot(q1.T, xv) * t1
                if 0 > vyr1:
                    c1 = testumkp(c, xv, q1, t1)
                    q1 = q1 + c1*c*t1*xv
                vyr2 = np.dot(q2.T, xv) * t2
                if 0 > vyr2:
                    c2 = testumkp(c, xv, q2, t2)
                    q2 = q2 + c2 * c * t2 * xv
                vyr3 = np.dot(q3.T, xv) * t3
                if 0 > vyr3:
                    c3 = testumkp(c, xv, q3, t3)
                    q3 = q3 + c3* c* t3 * xv
        chyba = []
        for i in range(len(tridy)):
            for j in range(len(tridy[i])):
                xv = np.array([1, tridy[i][j][0], tridy[i][j][1]])
                t1 = 0
                t2 = 0
                t3 = 0
                if i == 0:
                    t1 = 1
                    t2 = -1
                    t3 = -1
                elif i == 1:
                    t1 = -1
                    t2 = 1
                    t3 = -1
                else:
                    t1 = -1
                    t2 = -1
                    t3 = 1
                vyr1 = np.dot(q1.T, xv) * t1
                if 0 > vyr1:
                    chyba.append(1)
                vyr2 = np.dot(q2.T, xv) * t2
                if 0 > vyr2:
                    chyba.append(1)
                vyr3 = np.dot(q3.T, xv) * t3
                if 0 > vyr3:
                    chyba.append(1)
        if 1 not in chyba:
            break
    return [q1, q2, q3, iterace]


def SGD(tridy, c):#SGD trenovani vstup: seznam seznamu, konstanta uceni vystup: nastaveni klasifikatoru
    w = np.random.randn(3, 2)
    b = np.random.randn(3)
    b = np.array([b])
    q = 0
    while True:
        E = 0
        for i in range(len(tridy)):
            for j in range(len(tridy[i])):
                if i == 0:
                    t1 = 1
                    t2 = -1
                    t3 = -1
                elif i == 1:
                    t1 = -1
                    t2 = 1
                    t3 = -1
                else:
                    t1 = -1
                    t2 = -1
                    t3 = 1
                u = np.array([[t1, t2, t3]])
                trans = np.array([tridy[i][j]])
                vys = np.transpose(np.dot(w, np.transpose(trans))) + b
                y = np.sign(vys)
                y = np.array(y)
                v = np.array(u - y)
                E = E + 0.5 *(np.dot(v, v.T))
                w = w + c * (v.T.dot(trans))
                b = b + c * v
        q += 1
        if E == 0:
            break
    print("Pocet epoch", q)
    return [w, b]

def BGD(tridy, c): #BGD trenovani vstup: seznam seznamu, konstanta uceni vystup: nastaveni klasifikatoru
    w = np.random.randn(3, 2)
    b = np.random.randn(3)
    b = np.array([b])
    q = 0
    while True:
        wep = np.zeros((3, 2))
        bep = np.zeros((1, 3))
        E = 0
        for i in range(len(tridy)):
            for j in range(len(tridy[i])):
                if i == 0:
                    t1 = 1
                    t2 = -1
                    t3 = -1
                elif i == 1:
                    t1 = -1
                    t2 = 1
                    t3 = -1
                else:
                    t1 = -1
                    t2 = -1
                    t3 = 1
                u = np.array([[t1, t2, t3]])
                trans = np.array([tridy[i][j]])
                vys = np.transpose(np.dot(w, np.transpose(trans))) + b
                y = np.sign(vys)
                y = np.array(y)
                v = np.array(u - y)
                E = E + 0.5 *(np.dot(v, v.T))
                wep += c * (v.T.dot(trans))
                bep += c * v
        q += 1
        #print("Aktualni chyba", E)
        if E == 0:
            break
        w = w + wep#/(len(tridy[1]) + len(tridy[0]) + len(tridy[2]))
        b = b + bep#/(len(tridy[1]) + len(tridy[0]) + len(tridy[2]))
    print("Pocet epoch", q)
    return [w, b]


def ohodneur(w, b, x): # ohodnoci bod v prostoru vstup: wahy, prachy, bod vystup: vektor
    x = np.array([x])
    vys = np.transpose(np.dot(w, np.transpose(x))) + b
    y = np.sign(vys)
    return y


def ohodpros(klasif, x):# ohodnoceni rosenblata
    vys = []
    xv = np.array([1, x[0], x[1]])
    for i in range(3):
        vyr = np.dot(klasif[i].T, xv)
        if vyr>0:
            vys.append(1)
        else:
            vys.append(0)
    return vys


def rastr(): # metoda dela sit
    vektors = []
    X = []
    Y = []
    x = np.linspace(-21, 18, num=80)
    y = np.linspace(-24, 22, num=60)
    for i in range(len(x)):
        for j in range(len(y)):
            X.append(x[i])
            Y.append(y[j])
            vektors.append(np.array([x[i], y[j]]))
    return [vektors, X, Y]


def ohodbay(tridy): # vykresli bayssuv klasifikator
    for i in range(len(tridy)):
        mnoziny = nacteny(tridy[i])
        plt.scatter(mnoziny[0], mnoziny[1])
    sit = rastr()
    for i in range(len(sit[0])):
        vys = pravbayes(tridy, sit[0][i])
        if sum(vys) > 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='0.75')
        elif vys[0] == 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='b')
        elif vys[1] == 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='y')
        elif vys[2] == 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='g')
    plt.title("Bayesův klasifikátor")
    plt.show()


def ohodjsou(tridy):# graf jeden soused
    for i in range(len(tridy)):
        mnoziny = nacteny(tridy[i])
        plt.scatter(mnoziny[0], mnoziny[1])
    sit = rastr()
    for i in range(len(sit[0])):
        vys = jedns(tridy, sit[0][i])
        if sum(vys) != 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='0.75')
        elif vys[0] == 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='b')
        elif vys[1] == 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='y')
        elif vys[2] == 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='g')
    plt.title("klasifikátor podle jedného nejbližšího souseda")
    plt.show()


def ohodkv(tridy):#graf vektorove kvantizace
    for i in range(len(tridy)):
        mnoziny = nacteny(tridy[i])
        plt.scatter(mnoziny[0], mnoziny[1])
    sit = rastr()
    vys1 = kvant(tridy, sit[0][0])
    for i in range(len(sit[0])):
        vys = kvant(tridy, sit[0][i])
        if sum(vys[0]) > 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='0.75')
        elif vys[0][0] == 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='b')
        elif vys[0][1] == 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='y')
        elif vys[0][2] == 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='g')
    for i in range(len(vys1[1])):
        plt.plot(vys1[1][i][0], vys1[1][i][1], '.', color='black')
    plt.title("vektorova kvantizace")
    plt.show()


def ohoddsou(tridy):# graf podle dvou nejblissich sousedu
    for i in range(len(tridy)):
        mnoziny = nacteny(tridy[i])
        plt.scatter(mnoziny[0], mnoziny[1])
    sit = rastr()
    for i in range(len(sit[0])):
        vys = dvs(tridy, sit[0][i])
        if sum(vys) != 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='0.75')
        elif vys[0] == 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='b')
        elif vys[1] == 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='y')
        elif vys[2] == 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='g')
    plt.title("klasifikátor podle dvou nejbližších sousedů")
    plt.show()


def grafras(tridy):#graf rosenblat
    x = trenros(tridy)
    for i in range(len(tridy)):
        mnoziny = nacteny(tridy[i])
        plt.scatter(mnoziny[0], mnoziny[1])
    sit = rastr()
    for i in range(len(sit[0])):
        vys = ohodpros([x[0], x[1], x[2]], sit[0][i])
        if sum(vys) != 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='0.75')
        elif vys[0] == 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='b')
        elif vys[1] == 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='y')
        elif vys[2] == 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='g')
    plt.title("Rosenblattův algoritmus")
    plt.show()
    print("Počet iteraci", x[3])


def grafumkp(tridy, c):# graf UMKP
    x = umkp(tridy, c)
    for i in range(len(tridy)):
        mnoziny = nacteny(tridy[i])
        plt.scatter(mnoziny[0], mnoziny[1])
    sit = rastr()
    for i in range(len(sit[0])):
        vys = ohodpros([x[0], x[1], x[2]], sit[0][i])
        if sum(vys) != 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='0.75')
        elif vys[0] == 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='b')
        elif vys[1] == 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='y')
        elif vys[2] == 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='g')
    plt.title("Upravená metoda konstantních přírůstků")
    plt.show()
    print("Počet iteraci", x[3])


def grafneur(tridy, c):#graf SGD
    x = SGD(tridy, c)
    for i in range(len(tridy)):
        mnoziny = nacteny(tridy[i])
        plt.scatter(mnoziny[0], mnoziny[1])
    sit = rastr()
    for i in range(len(sit[0])):
        vys = np.squeeze(ohodneur(x[0], x[1],  sit[0][i]))
        if sum(vys) != -1:
            plt.plot(sit[1][i], sit[2][i], '.', color='0.75')
        elif vys[0] == 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='b')
        elif vys[1] == 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='y')
        elif vys[2] == 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='g')
    plt.title("SGD")
    plt.show()


def grafneur2(tridy, c):#graf BGD
    x = BGD(tridy, c)
    for i in range(len(tridy)):
        mnoziny = nacteny(tridy[i])
        plt.scatter(mnoziny[0], mnoziny[1])
    sit = rastr()
    for i in range(len(sit[0])):
        vys = np.squeeze(ohodneur(x[0], x[1],  sit[0][i]))
        if sum(vys) != -1:
            plt.plot(sit[1][i], sit[2][i], '.', color='0.75')
        elif vys[0] == 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='b')
        elif vys[1] == 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='y')
        elif vys[2] == 1:
            plt.plot(sit[1][i], sit[2][i], '.', color='g')
    plt.title("BGD")
    plt.show()


body = otevrit('data.txt')
vzd = shlukhlad(body)
plt.plot(vzd, '.')
plt.show()
vystup2 = mapa(body, 175)
for i in range(len(vystup2)):
    mnoziny = nacteny(vystup2[i])
    plt.scatter(mnoziny[0], mnoziny[1])
plt.title("mapa")
plt.show()
vystup = maxmin(body, 0.8)
for i in range(len(vystup)):
    mnoziny = nacteny(vystup[i])
    plt.scatter(mnoziny[0], mnoziny[1])
plt.title('maxmin')
plt.show()
vystup1 = kmeans(body, 3)
print("Celkova hodanota kriteria kmeans primim delenim:", vystup1[1])
for i in range(3):
     mnoziny = nacteny(vystup1[0][i])
     plt.scatter(mnoziny[0], mnoziny[1])
plt.title("kmeans")
plt.show()
vystup3 = kmeannebin(body)
for i in range(3):
     mnoziny = nacteny(vystup3[i])
     plt.scatter(mnoziny[0], mnoziny[1])
plt.title("kmeansbin")
plt.show()
res = optim(vystup1[0])
for i in range(len(res)):
    mnoziny = nacteny(res[i])
    plt.scatter(mnoziny[0], mnoziny[1])
plt.title('iterativni optimalizace')
plt.show()
ohodbay(vystup1[0])
ohodkv(vystup1[0])
ohodjsou(vystup1[0])
ohoddsou(vystup1[0])
grafras(vystup1[0])
grafumkp(vystup1[0], 0.7)
grafneur(vystup1[0], 0.7)
grafneur2(vystup1[0], 1)
