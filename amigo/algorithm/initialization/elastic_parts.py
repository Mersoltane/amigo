"""The split of a residual into the elastic pair that the barrier centers."""

import numpy as np


def elastic_split(r, mu, rho):
    """Elastic pair p - n = r that minimizes rho (p + n) - mu log p - mu log n.

    With a = mu / rho and s = sqrt(r^2 + a^2) the pair has the product
    p n = mu (a + s) / (2 rho), which gives each root without cancellation.
    """
    a = mu / rho
    s = np.sqrt(r * r + a * a)
    pn = mu * (a + s) / (2.0 * rho)
    p_pos = 0.5 * (a + r + s)
    n_neg = 0.5 * (a - r + s)
    p = np.where(r >= 0.0, p_pos, pn / n_neg)
    n = np.where(r >= 0.0, pn / p_pos, n_neg)
    return p, n
