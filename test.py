import random 

def crossover(p1, p2):
    size = min(len(p1), len(p2))
    c1, c2 = ['x'] * size, ['x'] * size

    cutIdx = set()
    while len(cutIdx) < 2:
        cutIdx |= {random.randint(1, size - 1)}
    cutIdx = list(cutIdx)
    cutIdx1, cutIdx2 = cutIdx[0], cutIdx[1]
    cutIdx1, cutIdx2 = min(cutIdx1, cutIdx2), max(cutIdx1, cutIdx2)
    c1[cutIdx1:cutIdx2] = p2[cutIdx1:cutIdx2]
    c2[cutIdx1:cutIdx2] = p1[cutIdx1:cutIdx2]

    p1_dict={}
    p2_dict={}
    for i in range(size):
        if c1[i] == 'x' or c2[i] == 'x':
            if p1[i] not in c1:
                c1[i] = p1[i]
            if p2[i] not in c2:
                c2[i] = p2[i]

        p1_dict[p1[i]] = p2[i]
        p2_dict[p2[i]] = p1[i]

    for i in range(size):
        if c1[i] == 'x':
            new_gen = p1[i]
            while True:
                new_gen = p2_dict[new_gen]
                if new_gen not in c1:
                    c1[i] = new_gen
                    break
        if c2[i] == 'x':
            new_gen = p1[i]
            while True:
                new_gen = p1_dict[new_gen]
                if new_gen not in c2:
                    c2[i] = new_gen
                    break

    return c1, c2

a = [1, 2, 3 ,4 , 5]
b = [3, 1, 5 ,2 , 4]
c = a[1:3]
c.reverse()
# print(crossover(a, b))
print(a[:1] + c + a[3:])