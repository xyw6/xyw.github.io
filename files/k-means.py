import numpy as np
X = np.array([[65, 60], [53, 60], [65, 62], [53, 64], [68, 63], [51, 57], [60, 51], [60, 80]])
init = np.array([[53, 60], [65, 60]])
def kmeans(points, initials):
    center, l = initials, True
    while l:
        ds = []
        for i in center:
            d = points - i
            d *= d
            d = np.sum(d,axis=1)
            ds.append(d)
        ds = np.array(ds).T
        C = [[] for i in range(len(center))]
        for i in range(len(points)):
            C[np.argmin(ds[i])].append(points[i])
        new_center = np.array([np.mean(c, axis=0) for c in C])
        if (new_center == center).all():
            l = False
        else:
            center = new_center
    return C, center
C,center = kmeans(X, init)
def sequential_k_means(points, initials):
    center, n, C = initials.copy(), [0] * len(initials), [[] for i in range(len(initials))]
    for p in points:
        d = center - p
        d *= d
        d = np.sum(d,axis=1)
        i = np.argmin(d)
        C[i].append(p)
        n[i] += 1
        center[i] = center[i] + (p - center[i]) / n[i]
    return C, center
C2,center2 = sequential_k_means(X, init)
for cc in [C, C2]:
    for c in cc:
        for i in range(len(c)):
            c[i] = (c[i][0], c[i][1])