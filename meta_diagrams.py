import numpy as np
from scipy.spatial.distance import squareform, pdist
from itertools import combinations
import time
import matplotlib.pyplot as plt
import copy


def get_lows(Rcols):
    lows = []
    for ind in range(len(Rcols)):
        if len(Rcols[ind]) > 0:
            lows.append(max(Rcols[ind]))
        else:
            lows.append(-1)
    return lows


def get_final_fvals(fvals):
    indices = []
    n = len(fvals)
    for i in range(n-1):
        if fvals[i] != fvals[i+1]:
            indices.append(i)
    indices.append(n-1)
    return indices


def compute_RU(D):
    R = D
    n = len(D)
    lows = np.zeros(n)
    for i in range(n):
        temp = np.array(R[:, i])
        if np.any(temp):
            lows[i] = np.max(np.where(temp))
        else:
            lows[i] = -1
    lows = np.array(lows)
    U = np.eye(n)
    for i in range(n):
        if lows[i] == -1:
            continue
        else:
            unred = True
        while unred:
            for j in range(i):
                if lows[i] == -1:
                    break
                if lows[i] == lows[j]:
                    R[:, i] = np.mod(R[:, i] + R[:, j], 2)
                    U[j, i] = 1
                    nz_loc = np.where(R[:, i])
                    if len(nz_loc[0]) == 0:
                        lows[i] = -1
                        continue
                    else:
                        lows[i] = np.max(np.where(R[:, i]))

            unred = False
            for j in range(i-1):
                if lows[i] == lows[j] and lows[i] > -1:
                    unred = True
    return R, U, lows


def compute_pairs(R, U, lows):
    n = len(lows)
    pos_inds = np.zeros(n)
    paired = []
    intervals = []
    for i in range(len(lows)):
        if lows[i] > -1:
            intervals.append([lows[i], i])
            pos_inds[int(lows[i])] = 1
            paired.append(int(lows[i]))
            paired.append(int(i))
    for j in range(n):
        if j not in paired:
            intervals.append([j, n])
            pos_inds[j] = 1
    intervals = np.array(intervals)
    intervals = intervals[intervals[:, 0].argsort()]
    return intervals, pos_inds


def transposition(Rrows, Rcols, Urows, Ucols, i, intervals, pos_inds, dims):
    dims[i], dims[i + 1] = dims[i + 1], dims[i]
    if dims[i] != dims[i+1]:
        Rrows, Rcols, Urows, Ucols, intervals, pos_inds = diff_dim(Rrows, Rcols, Urows, Ucols, i, intervals, pos_inds)
        return Rrows, Rcols, Urows, Ucols, intervals, pos_inds, dims
    # Case 1:
    if pos_inds[i] == 1 and pos_inds[i+1] == 1:
        Rrows, Rcols, Urows, Ucols, intervals, pos_inds = tcase_1(Rrows, Rcols, Urows, Ucols, i, intervals, pos_inds)
        return Rrows, Rcols, Urows, Ucols, intervals, pos_inds, dims
    # Case 2:
    if pos_inds[i] == 0 and pos_inds[i+1] == 0:
        Rrows, Rcols, Urows, Ucols, intervals, pos_inds = tcase_2(Rrows, Rcols, Urows, Ucols, i, intervals, pos_inds)
        return Rrows, Rcols, Urows, Ucols, intervals, pos_inds, dims
    # Case 3:
    if pos_inds[i] == 0 and pos_inds[i+1] == 1:
        Rrows, Rcols, Urows, Ucols, intervals, pos_inds = tcase_3(Rrows, Rcols, Urows, Ucols, i, intervals, pos_inds)
        return Rrows, Rcols, Urows, Ucols, intervals, pos_inds, dims
    # Case 4:
    if pos_inds[i] == 1 and pos_inds[i+1] == 0:
        Rrows, Rcols, Urows, Ucols, intervals, pos_inds = tcase_4(Rrows, Rcols, Urows, Ucols, i, intervals, pos_inds)
        return Rrows, Rcols, Urows, Ucols, intervals, pos_inds, dims


def diff_dim(Rrows, Rcols, Urows, Ucols, i, intervals, pos_inds):
    old_i = np.where(intervals == i)
    old_i = int(old_i[0])
    old_ib = intervals[old_i, 0]
    old_id = intervals[old_i, 1]
    old_ip1 = np.where(intervals == i+1)
    old_ip1 = int(old_ip1[0])
    old_ip1b = intervals[old_ip1, 0]
    old_ip1d = intervals[old_ip1, 1]
    if old_i != old_ip1:
        Rrows, Rcols, Urows, Ucols = my_swaps(Rrows, Rcols, Urows, Ucols, i)
        if pos_inds[i] == 1:
            intervals[old_i] = np.array([i+1, old_id])
        elif pos_inds[i] == 0:
            intervals[old_i] = np.array([old_ib, i+1])
        if pos_inds[i+1] == 1:
            intervals[old_ip1] = np.array([i, old_ip1d])
        elif pos_inds[i+1] == 0:
            intervals[old_ip1] = np.array([old_ip1b, i])
    pi = pos_inds[i]
    pip1 = pos_inds[i+1]
    pos_inds[i], pos_inds[i+1] = pip1, pi

    return Rrows, Rcols, Urows, Ucols, intervals, pos_inds


def my_swaps(Rrows, Rcols, Urows, Ucols, i):
    Rcols[i], Rcols[i+1] = Rcols[i+1], Rcols[i]
    for j in range(len(Rcols)):
        if i in Rcols[j] and i+1 not in Rcols[j]:
            Rcols[j].remove(i)
            Rcols[j].add(i+1)
        elif i+1 in Rcols[j] and i not in Rcols[j]:
            Rcols[j].remove(i+1)
            Rcols[j].add(i)
    Urows[i], Urows[i+1] = Urows[i+1], Urows[i]
    Ucols[i], Ucols[i+1] = Ucols[i+1], Ucols[i]

    for j in range(len(Urows)):
        if i in Urows[j] and i+1 not in Urows[j]:
            Urows[j].remove(i)
            Urows[j].add(i+1)
        elif i+1 in Urows[j] and i not in Urows[j]:
            Urows[j].remove(i+1)
            Urows[j].add(i)
    return Rrows, Rcols, Urows, Ucols


def tcase_1(Rrows, Rcols, Urows, Ucols, i, intervals, pos_inds):
    maxn = len(Rcols) - 1
    lows = get_lows(Rcols)
    orig_lows = lows
    old_i = np.where(intervals == i)
    old_i = int(old_i[0])
    old_ib = intervals[old_i, 0]
    old_id = intervals[old_i, 1]
    old_ip1 = np.where(intervals == i+1)
    old_ip1 = int(old_ip1[0])
    old_ip1b = intervals[old_ip1, 0]
    old_ip1d = intervals[old_ip1, 1]
    oRcols = copy.deepcopy(Rcols)

    Rrows, Rcols, Urows, Ucols = my_swaps(Rrows, Rcols, Urows, Ucols, i)
    old_intervals = np.copy(intervals)
    intervals[old_i], intervals[old_ip1] = np.array([i+1, old_id]), np.array([i, old_ip1d])
    if i in Urows[i+1]:
        Urows[i+1].remove(i)
    k, l = np.inf, np.inf

    for j in range(i+2, len(Rrows)):
        if orig_lows[j] == i:
            k = j
        if orig_lows[j] == i+1:
            l = j
    if k == np.inf and l == np.inf:
        return Rrows, Rcols, Urows, Ucols, intervals, pos_inds
    # Case 1.1
    if k < np.inf and l < np.inf:
        if i in oRcols[l]:
            # Case 1.1.1
            if k < l:
                Rcols[l] = Rcols[l] ^ Rcols[k]
                Urows[k] = Urows[l] ^ Urows[k]
            # Case 1.1.2
            elif l < k:
                Rcols[k] = Rcols[l] ^ Rcols[k]
                Urows[l] = Urows[l] ^ Urows[k]
                intervals[old_i], intervals[old_ip1] = old_intervals[old_i], old_intervals[old_ip1]
        # Case 1.2 empty as only change is the swaps already performed.
    elif l < np.inf:
        if i in oRcols[l]:
            #Rcols[k] = Rcols[l] ^ Rcols[k]
            #Urows[l] = Urows[l] ^ Urows[k]
            intervals[old_i], intervals[old_ip1] = old_intervals[old_i], old_intervals[old_ip1]

    return Rrows, Rcols, Urows, Ucols, intervals, pos_inds


def tcase_2(Rrows, Rcols, Urows, Ucols, i, intervals, pos_inds):
    orig_lows = get_lows(Rcols)
    old_i = np.where(intervals == i)
    old_i = int(old_i[0])
    old_ib = intervals[old_i, 0]
    old_id = intervals[old_i, 1]
    old_ip1 = np.where(intervals == i+1)
    old_ip1 = int(old_ip1[0])
    old_ip1b = intervals[old_ip1, 0]
    old_ip1d = intervals[old_ip1, 1]
    old_intervals = np.copy(intervals)
    intervals[old_i], intervals[old_ip1] = np.array([old_ib, i+1]), np.array([old_ip1b, i])
    # Case 2.2
    if i+1 not in Urows[i]:
        Rrows, Rcols, Urows, Ucols = my_swaps(Rrows, Rcols, Urows, Ucols, i)
        return Rrows, Rcols, Urows, Ucols, intervals, pos_inds
    else:
        Urows[i] = Urows[i] ^ Urows[i+1]
        Rcols[i+1] = Rcols[i] ^ Rcols[i+1]

        # Case 2.1.1
        if orig_lows[i] < orig_lows[i+1]:
            Rrows, Rcols, Urows, Ucols = my_swaps(Rrows, Rcols, Urows, Ucols, i)
            return Rrows, Rcols, Urows, Ucols, intervals, pos_inds

        # Case 2.1.2
        elif orig_lows[i+1] < orig_lows[i]:
            Rrows, Rcols, Urows, Ucols = my_swaps(Rrows, Rcols, Urows, Ucols, i)
            Rcols[i+1] = Rcols[i+1] ^ Rcols[i]
            Urows[i] = Urows[i] ^ Urows[i+1]
            intervals[old_i], intervals[old_ip1] = old_intervals[old_ip1], old_intervals[old_i]
            return Rrows, Rcols, Urows, Ucols, intervals, pos_inds


def tcase_3(Rrows, Rcols, Urows, Ucols, i, intervals, pos_inds):
    old_i = np.where(intervals == i)
    old_i = int(old_i[0])
    old_ib = intervals[old_i, 0]
    old_id = intervals[old_i, 1]
    old_ip1 = np.where(intervals == i+1)
    old_ip1 = int(old_ip1[0])
    old_ip1b = intervals[old_ip1, 0]
    old_ip1d = intervals[old_ip1, 1]
    old_intervals = np.copy(intervals)
    intervals[old_i], intervals[old_ip1] = np.array([old_ib, i+1]), np.array([i, old_ip1d])
    # Case 3.2
    if i+1 not in Urows[i]:
        Rrows, Rcols, Urows, Ucols = my_swaps(Rrows, Rcols, Urows, Ucols, i)
        pos_inds[i], pos_inds[i + 1] = 1, 0
        return Rrows, Rcols, Urows, Ucols, intervals, pos_inds
    # Case 3.1
    if i+1 in Urows[i]:
        Urows[i] = Urows[i] ^ Urows[i+1]
        Rcols[i+1] = Rcols[i+1] ^ Rcols[i]
        Rrows, Rcols, Urows, Ucols = my_swaps(Rrows, Rcols, Urows, Ucols, i)
        Rcols[i+1] = Rcols[i+1] ^ Rcols[i]
        Urows[i] = Urows[i] ^ Urows[i+1]
        intervals[old_i], intervals[old_ip1] = old_intervals[old_i], old_intervals[old_ip1]
        pos_inds[i], pos_inds[i + 1] = 0, 1
        return Rrows, Rcols, Urows, Ucols, intervals, pos_inds


def tcase_4(Rrows, Rcols, Urows, Ucols, i, intervals, pos_inds):
    pos_inds[i], pos_inds[i+1] = 0, 1
    if i+1 in Urows[i]:
        Urows[i].remove(i+1)
    Rrows, Rcols, Urows, Ucols = my_swaps(Rrows, Rcols, Urows, Ucols, i)
    old_i = np.where(intervals == i)
    old_i = int(old_i[0])
    old_ib = intervals[old_i, 0]
    old_id = intervals[old_i, 1]
    old_ip1 = np.where(intervals == i+1)
    old_ip1 = int(old_ip1[0])
    old_ip1b = intervals[old_ip1, 0]
    old_ip1d = intervals[old_ip1, 1]
    intervals[old_i], intervals[old_ip1] = np.array([i+1, old_id]), np.array([old_ip1b, i])
    return Rrows, Rcols, Urows, Ucols, intervals, pos_inds


def get_filt_values(dx, fx, dim, thresh=np.inf):
    n = len(fx)
    s = range(n)
    simplices = []
    for i in range(dim+1):
        simplices += list(map(list, combinations(s, i+1)))
    vr_values = []
    fx_values = []
    keep_inds = []
    for i in range(len(simplices)):
        indices = simplices[i]
        fxval = np.max(fx[indices])
        fx_values.append(fxval)

        d2_indices = np.ix_(indices, indices)
        vrval = np.max(dx[d2_indices])
        vr_values.append(vrval)
        if vrval <= thresh:
            keep_inds.append(i)
    simplices = [simplices[ind] for ind in keep_inds]
    fx_values = [fx_values[ind] for ind in keep_inds]
    vr_values = [vr_values[ind] for ind in keep_inds]
    sort_inds = np.lexsort((np.array(vr_values), np.array(fx_values)))
    simplices = [simplices[sort_inds[ind]] for ind in range(len(sort_inds))]
    fx_values = [fx_values[sort_inds[ind]] for ind in range(len(sort_inds))]
    vr_values = [vr_values[sort_inds[ind]] for ind in range(len(sort_inds))]

    return simplices, fx_values, vr_values


def get_boundary(simplices):
    D = np.zeros((len(simplices), len(simplices)))
    for i in range(len(simplices)):
        s = simplices[i]
        sdim = len(simplices[i])
        if sdim == 1:
            continue
        faces = list(map(list, combinations(s, sdim - 1)))
        for face in faces:
            face_ind = simplices.index(face)
            D[face_ind, i] = 1
    return D


def get_true_intervals(mrk_temp, orig_simplices, vrvals):
    n = len(vrvals)
    for i in range(len(mrk_temp)):
        if mrk_temp[i][0] == n:
            mrk_temp[i][0] = np.inf
        else:
            mrk_temp[i][0] = vrvals[orig_simplices[int(mrk_temp[i][0])]]
        if mrk_temp[i][1] == n:
            mrk_temp[i][1] = np.inf
        else:
            mrk_temp[i][1] = vrvals[orig_simplices[int(mrk_temp[i][1])]]
    return mrk_temp


def get_filt_loc(vrvals, tagged=False):
    cvrval = vrvals[-1]
    svrvals = np.sort(vrvals)
    inds = np.where(svrvals == cvrval)
    return np.max(inds[0])


def mrk(simplices, fxvals, vrvals):
    orig_simplices = [i for i in range(len(simplices)+1)]
    dims = [len(simplices[i]) for i in range(len(simplices))]

    n = len(fxvals)
    mrk = []
    D = get_boundary(simplices)
    print(f"Initial boundary computation along gamma_1 complete.")

    st2 = time.time()
    R, U, lows = compute_RU(D)
    print(f"Boundary reduction along gamma_1 complete, moving to transpositions")

    intervals, pos_inds = compute_pairs(R, U, lows)
    simplices.append([-1])
    vrvals.append(np.inf)
    dims.append(np.inf)

    Rrows = [set(np.where(R[i] > 0)[0]) for i in range(len(R))]
    Rcols = [set(np.where(R[:, i] > 0)[0]) for i in range(len(R))]

    Urows = [set(np.where(U[i] > 0)[0]) for i in range(len(U))]
    Ucols = [set(np.where(U[:, i] > 0)[0]) for i in range(len(U))]

    mrk_temp = np.copy(intervals)
    for i in range(len(mrk_temp)):
        if mrk_temp[i][0] > 0:
            mrk_temp[i][0] = n
        if mrk_temp[i][1] > 0:
            mrk_temp[i][1] = n
    mrk_real = get_true_intervals(mrk_temp, orig_simplices, vrvals)
    mrk.append(mrk_real)

    for i in range(1, n):
        new_loc = get_filt_loc(vrvals[:i+1])
        print(f"\non iteration {i} of meta-rank, needing to do {i-new_loc} transpositions")
        print(f"the simplex moving forwards is {simplices[orig_simplices[i]]}")

        for j in range(i, new_loc, -1):
            Rrows, Rcols, Urows, Ucols, intervals, pos_inds, dims = transposition(Rrows, Rcols, Urows, Ucols, j-1, intervals, pos_inds, dims)
            orig_simplices[j-1], orig_simplices[j] = orig_simplices[j], orig_simplices[j-1]

        mrk_temp = np.copy(intervals)
        for k in range(len(mrk_temp)):
            if mrk_temp[k][0] > i:
                mrk_temp[k][0] = n
            if mrk_temp[k][1] > i:
                mrk_temp[k][1] = n
        mrk_real = get_true_intervals(mrk_temp, orig_simplices, vrvals)
        mrk.append(mrk_real)
        orig_simplices.append(-1)
        fdims = [dims[int(intervals[k][0])] for k in range(len(intervals))]
    return mrk, fdims


def get_intersect(int1, int2):
    b1, d1 = float(int1[0]), float(int1[1])
    b2, d2 = float(int2[0]), float(int2[1])
    if b1 >= d2:
        return [b1, b1]
    elif b2 >= d1:
        return [b2, b2]
    else:
        return list([np.maximum(b1, b2), np.minimum(d1, d2)])


def get_fullmrk(pmrk):
    fmrk = [[[[pmrk[0][ind][0], pmrk[0][ind][1]] for ind in range(len(pmrk[0]))]]]
    for i in range(1, len(pmrk)):
        cmrk = [[[pmrk[i][ind][0], pmrk[i][ind][1]] for ind in range(len(pmrk[i]))]]
        for j in range(1, i+1):
            tmrk = []
            for k in range(len(pmrk[i])):
                tmrk.append(get_intersect(pmrk[i][k], fmrk[-1][i-j][k]))
            cmrk.append(tmrk)
        cmrk.reverse()
        fmrk.append(cmrk)
    return fmrk


def get_mdgm(fmrk, b, d):
    n = len(fmrk) - 1
    num_ints = len(fmrk[0][0])
    mdgmp = []
    mdgmn = []

    if b == 0 and d == n:
        return np.array(fmrk[d][b]), np.array([[np.inf, np.inf] for k in range(num_ints)])
    elif b == 0:
        mrkbd = np.around(fmrk[d][b], decimals=8)
        mrkbd1 = np.around(fmrk[d + 1][b], decimals=8)
        for k in range(num_ints):
            if np.all(mrkbd[k] == mrkbd1[k]):
                mdgmp.append([np.inf, np.inf])
                mdgmn.append([np.inf, np.inf])
            else:
                mdgmp.append(mrkbd[k])
                mdgmn.append(mrkbd1[k])
    elif d == n:
        mrkbd = np.around(fmrk[d][b], decimals=8)
        mrkb1d = np.around(fmrk[d][b - 1], decimals=8)
        for k in range(num_ints):
            if np.all(mrkbd[k] == mrkb1d[k]):
                mdgmp.append([np.inf, np.inf])
                mdgmn.append([np.inf, np.inf])
            else:
                mdgmp.append(mrkbd[k])
                mdgmn.append(mrkb1d[k])
    elif b > 0 and d < n:
        mrkbd = np.around(fmrk[d][b], decimals=8)
        mrkb1d = np.around(fmrk[d][b - 1], decimals=8)
        mrkbd1 = np.around(fmrk[d + 1][b], decimals=8)
        mrkb1d1 = np.around(fmrk[d + 1][b - 1], decimals=8)
        for k in range(num_ints):
            if np.all(mrkbd[k] == mrkb1d[k]) or np.all(mrkbd[k] == mrkbd1[k]):
                mdgmp.append([np.inf, np.inf])
                mdgmn.append([np.inf, np.inf])
            elif np.all(mrkb1d1[k] == mrkb1d[k]):
                mdgmp.append(mrkbd[k])
                mdgmn.append(mrkbd1[k])
            elif np.all(mrkb1d1[k] == mrkbd1[k]):
                mdgmp.append(mrkbd[k])
                mdgmn.append(mrkb1d[k])
            else:
                mdgmp.append([mrkbd[k], mrkb1d1[k]])
                mdgmn.append([mrkbd1[k], mrkb1d[k]])

    return np.array(mdgmp), np.array(mdgmn)


if __name__ == '__main__':
    # input the data as a point cloud
    point_cloud_file = 'fig8pts.txt'
    x = np.loadtxt(point_cloud_file)
    dx = pdist(x)
    dx = squareform(dx, 'tomatrix')
    dx = np.around(dx, decimals=6)

    # define the function for a function-Rips bifiltration. This is the height function
    fx = np.around(np.array([pt[1] for pt in x]), decimals=6)

    # define the maximal dimension of simplices to include
    homdim = 2

    # define a threshold for Vietoris-Rips values of simplices to include
    threshold = np.max(dx)/2

    # compute all simplices, and their corresponding function and Vietoris-Rips values, ordered lexicographically
    simplices, fxvals, vrvals = get_filt_values(dx, fx, homdim, thresh=threshold)
    print(f"the number of dim0 simplices = {np.sum(np.array([1 for i in range(len(simplices)) if len(simplices[i]) == 1]))}")
    print(f"the number of dim1 simplices = {np.sum(np.array([1 for i in range(len(simplices)) if len(simplices[i]) == 2]))}")
    print(f"the number of dim2 simplices = {np.sum(np.array([1 for i in range(len(simplices)) if len(simplices[i]) == 3]))}")
    print(f"the total number of simplices is {len(simplices)}")

    # compute the meta-rank
    pre_mrk, dims = mrk(simplices, fxvals, vrvals)
    dims = [a-1 for a in dims]

    final_fvals = get_final_fvals(fxvals)
    post_mrk = [pre_mrk[index] for index in final_fvals]
    fmrk = get_fullmrk(post_mrk)
    for j in range(len(fmrk)):
        for k in range(len(fmrk[j])):
            print(f"mrk_M([{fxvals[final_fvals[k]]}, {fxvals[final_fvals[j]]}]) is {fmrk[j][k]}")

    fpairs = []
    for i in range(len(final_fvals)):
        for j in range(i, len(final_fvals)):
            fpairs.append([final_fvals[i], final_fvals[j]])
    fpairs = np.array(fpairs)
    rfpairs = np.array([[fxvals[rval[0]], fxvals[rval[1]]] for rval in fpairs])

    fmdgmp = []
    fmdgmn = []
    for i in range(len(fmrk)):
        rowmdgmp = []
        rowmdgmn = []
        for j in range(i, -1, -1):
            mdgmp, mdgmn = get_mdgm(fmrk, j, i)
            rowmdgmp.append(mdgmp)
            rowmdgmn.append(mdgmn)
        rowmdgmp.reverse()
        rowmdgmn.reverse()
        fmdgmp.append(rowmdgmp)
        fmdgmn.append(rowmdgmn)

    nzpts = []
    for i in range(len(fmdgmp)):
        for j in range(len(fmdgmp[i])):
            for k in range(len(fmdgmp[i][j])):
                interval = fmdgmp[i][j][k]
                if interval[0] < np.inf:
                    if interval[1] - interval[0] > 0.001 and dims[k] <= homdim:
                        nzpts.append([i, j])
                        break
    nzpts = np.array(nzpts)
    pfp1 = [fxvals[int(final_fvals[ind])] for ind in nzpts[:, 1]]
    pfp2 = [fxvals[int(final_fvals[ind])] for ind in nzpts[:, 0]]
    rfpairs = np.vstack((pfp1, pfp2)).T
    fig, axs = plt.subplots(1, 2)
    mng = plt.get_current_fig_manager()

    axs[0].axline((0, 0), slope=1, zorder=1)
    axs[0].scatter(pfp1, pfp2, s=25, c='k', zorder=2)
    axs[0].set_title("Domain intervals of mdgm")
    axs[0].axis('equal')
    ylims = axs[0].get_ylim()
    xlims = axs[0].get_xlim()
    mkpossibles = ["s", "D", "v"]
    colorpossibles = ["r", "b"]
    num_clicks = 0

    while True:
        mdgmpt = plt.ginput(n=1, timeout=0)
        if num_clicks >= 1:
            axs[0].scatter(pbpt, pdpt, c='k', zorder=3)
        num_clicks += 1

        mpt = np.array([mdgmpt[0][0], mdgmpt[0][1]])
        dlist = np.array([np.abs(a[0]-mpt[0])**2 + np.abs(a[1]-mpt[1])**2 for a in rfpairs])
        idx = dlist.argmin()
        pbpt = rfpairs[idx][0]
        pdpt = rfpairs[idx][1]
        axs[0].scatter(pbpt, pdpt, c='r', zorder=3)
        axs[0].set(xlim=xlims, ylim=ylims)
        bind = nzpts[idx][1]
        dind = nzpts[idx][0]

        mdgmp = np.array(fmdgmp[dind][bind])
        mdgmn = np.array(fmdgmn[dind][bind])
        mycolors = []
        mks = []
        myx = []
        myy = []
        mydims = []
        myp = []

        axs[1].clear()
        plt.subplot(1, 2, 2)
        plocs = mdgmp[mdgmp < np.inf]
        if len(plocs) > 0:
            maxy = np.max(mdgmp[mdgmp < np.inf]) + 1
        else:
            maxy = np.max(vrvals[:-1])

        for i in range(len(mdgmp)):
            if mdgmp[i][0] is not list:
                if dims[i] >= homdim:
                    continue
                if mdgmp[i][0] == np.inf or mdgmp[i][0] == mdgmp[i][1]:
                    continue
                else:
                    if mdgmp[i][1] == np.inf:
                        mdgmp[i][1] = maxy
                    myx.append(mdgmp[i][0])
                    myy.append(mdgmp[i][1])
                    mycolors.append(colorpossibles[0])
                    mks.append(mkpossibles[dims[i]])
                    mydims.append(dims[i])
                    myp.append(1)
            else:
                if mdgmp[i][0][0] == np.inf or mdgmp[i][0][0] == mdgmp[i][0][1]:
                    continue
                else:
                    if dims[i] >= homdim:
                        continue
                    if mdgmp[i][0][1] == np.inf:
                        mdgmp[i][0][1] = maxy
                    myx.append(mdgmp[i][0][0])
                    myy.append(mdgmp[i][0][1])
                    mycolors.append(colorpossibles[0])
                    mks.append(mkpossibles[dims[i]])
                    mydims.append(dims[i])
                    myp.append(1)
                if mdgmp[i][1][0] == np.inf or mdgmp[i][1][0] == mdgmp[i][1][1]:
                    continue
                else:
                    if dims[i] >= homdim:
                        continue
                    if mdgmp[i][1][1] == np.inf:
                        mdgmp[i][1][1] = maxy
                    myx.append(mdgmp[i][1][0])
                    myy.append(mdgmp[i][1][1])
                    mycolors.append(colorpossibles[0])
                    mks.append(mkpossibles[dims[i]])
                    mydims.append(dims[i])
                    myp.append(1)
            if mdgmn[i][0] is not list:
                if mdgmn[i][0] == np.inf or mdgmn[i][0] == mdgmn[i][1]:
                    continue
                else:
                    if mdgmn[i][1] == np.inf:
                        mdgmn[i][1] = maxy
                    myx.append(mdgmn[i][0])
                    myy.append(mdgmn[i][1])
                    mycolors.append(colorpossibles[1])
                    mks.append(mkpossibles[dims[i]])
                    mydims.append(dims[i])
                    myp.append(0)
            else:
                if mdgmn[i][0][0] == np.inf or mdgmn[i][0][0] == mdgmn[i][0][1]:
                    continue
                else:
                    if mdgmn[i][0][1] == np.inf:
                        mdgmn[i][0][1] = maxy
                    myx.append(mdgmn[i][0][0])
                    myy.append(mdgmn[i][0][1])
                    mycolors.append(colorpossibles[1])
                    mks.append(mkpossibles[dims[i]])
                    mydims.append(dims[i])
                    myp.append(0)
                if mdgmn[i][1][0] == np.inf or mdgmn[i][1][0] == mdgmp[i][1][1]:
                    continue
                else:
                    if mdgmn[i][1][1] == np.inf:
                        mdgmn[i][1][1] = maxy
                    myx.append(mdgmn[i][1][0])
                    myy.append(mdgmn[i][1][1])
                    mycolors.append(colorpossibles[0])
                    mks.append(mkpossibles[dims[i]])
                    mydims.append(dims[i])
                    myp.append(0)

        for xp, yp, cp, m, mydim, mypv in zip(myx, myy, mycolors, mks, mydims, myp):
            if mypv == 1:
                lstr = f"Dim {mydim}, positive"
            else:
                lstr = f"Dim {mydim}, negative"
            axs[1].scatter(xp, yp, c=cp, s=50, marker=m, label=lstr, zorder=2)
        offset = maxy*.1
        axs[1].plot([-1*offset, maxy+offset], [maxy, maxy], "--", linewidth=1, c='k', label="inf", zorder=1)
        handles, labels = axs[1].get_legend_handles_labels()
        lbl_inds = sorted(range(len(labels)), key=labels.__getitem__)
        labels = [labels[ind] for ind in lbl_inds]
        handles = [handles[ind] for ind in lbl_inds]
        by_label = dict(zip(labels, handles))
        axs[1].legend(by_label.values(), by_label.keys(), loc='lower right')
        pbpt = np.around(pbpt, decimals=4)
        pdpt = np.around(pdpt, decimals=4)

        axs[1].set_title(f"mdgm[{pbpt}, {pdpt}]")

        axs[1].plot([-1*offset, maxy+offset], [-1*offset, maxy+offset], c='k')

        axs[1].axis(xmin=-1*offset, ymin=-1*offset, xmax=maxy+offset, ymax=maxy+offset)
