def normalize_matrix(a):
    # zero mean the vectors
    return a - a.mean(1)[:, None]

def corr_predictions(a, b):

    assert a.shape == b.shape, "a and b must have the same shape"
    n_el = a.shape[1]

    a = normalize_matrix(a)
    b = normalize_matrix(b)

    sigma_a = a.std(1)
    sigma_b = b.std(1)

    if sigma_a.size == 1:
        sigma_a = sigma_a.item()

    if sigma_b.size == 1:
        sigma_b = sigma_b.item()

    prod = sigma_a * sigma_b

    cov = (a * b).sum(1) / n_el
    return cov / prod
