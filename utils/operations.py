

def reorder(a, order):
    b = [None] * len(order)
    for x, k in zip(a, order):
        b[k] = x
    return b


def calc_weighted_average(a):
    val, weight = 0., 0
    for v, w in a:
        val += v * w
        weight += w
    return val / weight, weight