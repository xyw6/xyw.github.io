def distance(d_matrix, point, points):
    d = 0
    for p in points:
        if point > p:
            t = d_matrix[point][p]
        else:
            t = d_matrix[p][point]
        if t > d:
            d = t
    return d

def equal(A, B):
    if set(A[0]) == set(B[0]):
        return True
    if set(A[0]) == set(B[1]):
        return True
    return False

def clusting(d_matrix, A, B):
    delta = []
    #d, delta = [], []
    for p in A:
        #a, b = distance(d_matrix, p, A), distance(d_matrix, p, B)
        #d.append((a, b))
        delta.append(distance(d_matrix, p, A) - distance(d_matrix, p, B))
    md = max(delta)
    if md <= 0:
        return [[A, B]]
    else:
        result = []
        for i in range(len(A)):
            b = delta[i]
            if b == md:
                t = A[i]
                t1, t2 = A.copy(), B.copy()
                t1.remove(t)
                t2.append(t)
                result += clusting(d_matrix, t1, t2)
        els, final_result = set(), []
        for i in range(len(result)):
            if i not in els:
                final_result.append(result[i])
                for j in range(len(result)):
                    if j not in els:
                        if equal(result[i], result[j]):
                            els.add(j)
                
            
        return final_result
            

A =[[0], [11, 0], [5, 13, 0], [12, 2, 14, 0], [7, 17, 1, 18, 0], [13, 4, 15, 5, 20, 0], [9, 15, 12, 16, 15, 19, 0], [11, 20, 12, 21, 17, 22, 30, 0]]
for o in clusting(A, [0, 1, 2, 3, 4, 5, 6, 7], []):
    print(o)
    