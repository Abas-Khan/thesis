cdef unsigned long long fast_document_dbow_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
    np.uint32_t *cum_table_dis, unsigned long long cum_table_dis_len, np.uint32_t *cum_table_nondis, 
    unsigned long long cum_table_nondis_len, np.uint32_t *dis_indexes, np.uint32_t *nondis_indexes, 
    REAL_t *context_vectors, REAL_t *syn1neg, const int size, const np.uint32_t word_index,
    const np.uint32_t context_index,const np.uint32_t word_in_vocab, const np.uint32_t word2_in_vocab, const REAL_t alpha, REAL_t *work,
    unsigned long long next_random, int learn_context, int learn_hidden, REAL_t *context_locks,double max_min_optim, REAL_t *sample_select_dist, double sampling_pm, double objective_pm) nogil:

    cdef long long a
    cdef long long row1 = context_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label
    cdef np.uint32_t target_index
    cdef double sample_negative
    cdef int d

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            sample_negative = <double> rand()/(RAND_MAX+3.0)
            if sample_negative >= sampling_pm:
                target_index = dis_indexes[bisect_left(cum_table_dis, (next_random >> 16) % cum_table_dis[cum_table_dis_len-1], 0, cum_table_dis_len)]
            else:
                target_index = nondis_indexes[bisect_left(cum_table_nondis, (next_random >> 16) % cum_table_nondis[cum_table_nondis_len-1], 0, cum_table_nondis_len)]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0
        row2 = target_index * size
        f = our_dot(&size, &context_vectors[row1], &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        if learn_hidden:
            our_saxpy(&size, &g, &context_vectors[row1], &ONE, &syn1neg[row2], &ONE)
    if learn_context:
        our_saxpy(&size, &context_locks[context_index], work, &ONE, &context_vectors[row1], &ONE)

    return next_random