import numpy as np

def calculate_diversity(population):
    total_distance = 0
    number_of_pairs = 0

    for i in range(len(population) -1):
        for j in range(i, len(population)):
            total_distance += 1 - population[i].fast_similarity(population[j])
            number_of_pairs += 1

    return total_distance / number_of_pairs

def calculate_single_diversity(population, best_prog):
    total_distance = 0

    for i in range(len(population)):
        total_distance += 1 - population[i].similarity(best_prog)

    return total_distance / len(population)

def compute_locus_shannon_entropy(population):
    """Computes Mean Locus-wise Shannon Entropy across ALL loci in feature_map."""

    program_vectors = [p.similarity_vec for p in population]
    matrix = np.array(program_vectors, dtype=np.float64)  # (N_programs, K_features)
    n_programs, k_features = matrix.shape
    locus_sums = matrix.sum(axis=0)
    locus_entropy = np.zeros(k_features, dtype=np.float64)
    active_mask = locus_sums > 0

    # Normalize active columns to form probability diversity
    p = matrix[:, active_mask] / locus_sums[active_mask]

    # Compute Shannon entropy: -sum(p * log2(p))
    with np.errstate(divide='ignore', invalid='ignore'):
        log_p = np.where(p > 0, np.log2(p), 0.0)
        locus_entropy[active_mask] = -np.sum(p * log_p, axis=0)

    return locus_entropy

def compute_shannon_feature_coverage(population) -> float:
    """Calculates feature coverage as normalized Shannon entropy over global feature probabilities."""
    program_vectors = [p.similarity_vec for p in population]
    matrix = np.array(program_vectors, dtype=np.float64)  # (N_programs, K_features)
    n_programs, k_features = matrix.shape
    locus_sums = matrix.sum(axis=0)

    global_locus_prob = (
        locus_sums / locus_sums.sum() if  locus_sums.sum() > 0
        else np.zeros(k_features)
    )

    active_mask = global_locus_prob > 0
    p_active = global_locus_prob[active_mask]
    feature_entropy = -np.sum(p_active * np.log2(p_active))
    max_entropy = np.log2(k_features)
    global_feature_coverage = float(feature_entropy / max_entropy)

    return global_feature_coverage, global_locus_prob


# def reproduction(parent, parent_index, X, y, sample_weight, params, seed):
#     """Deep copy elite parent and assign updated reproduction genome."""
#     program = copy.deepcopy(parent)
#     program.parents = {
#         'method': 'Elite Copy',
#         'parent_idx': parent_index,
#         'parent_nodes': []
#     }
#
#     random_state = check_random_state(seed)
#     program._indices_state = None
#
#     max_samples = params['max_samples']
#     n_samples, n_features = X.shape
#     max_samples = int(max_samples * n_samples)
#
#     # Draw samples, using sample weights, and then fit
#     if sample_weight is None:
#         curr_sample_weight = np.ones((n_samples,))
#     else:
#         curr_sample_weight = sample_weight.copy()
#     oob_sample_weight = curr_sample_weight.copy()
#     indices, not_indices = program.get_all_indices(n_samples,
#                                                    max_samples,
#                                                    random_state)
#     curr_sample_weight[not_indices] = 0
#     oob_sample_weight[indices] = 0
#     program.raw_fitness_ = program.raw_fitness(X, y, curr_sample_weight)
#
#     if max_samples < n_samples:
#         # Calculate OOB fitness
#         program.oob_fitness_ = program.raw_fitness(X, y, oob_sample_weight)
#
#     return program