import numpy as np
import matplotlib.pyplot as plt
from itertools import product, chain, combinations
from copy import copy

from linear_algebra_inZ2 import loss_decoding_gausselim_fast


#############################################################
#### Functions to navigate between cells of a fusion lattice
#############################################################

def qbt_ix_from_coords(coords, L):
    coords_a = np.array(coords)
    if np.any(coords_a > (L - 1)) or np.any(coords_a < 0):
        raise ValueError('All coordinates need to be less than the size L')
    temp_ix = 0
    for dim_ix, coord_val in enumerate(coords):
        if coord_val > 0:
            temp_ix += coord_val * (L ** dim_ix)
    return temp_ix


def coords_from_qbt_ix(qbt_ix, L, num_dims):
    if qbt_ix > (L ** num_dims - 1):
        raise ValueError('stab_ix need to be less than L^num_dims-1')
    return np.array([int(qbt_ix / (L ** dim_ix)) % L for dim_ix in range(num_dims)])


def shifted_qbt_ix(qbt_ix, shift, shift_axis_ix, L, num_dims):
    #     Function to get the index of a cell obtained shifting an initial cell with 'cell_ix'
    #     by a integer (positive or negative) step 'shift' along an axis 'shift_axis'.
    if not isinstance(shift, int):
        raise ValueError('The parameter shift can only be an integer (positive or negative)')

    temp_coords = coords_from_qbt_ix(qbt_ix, L, num_dims)
    temp_coords[shift_axis_ix] = (temp_coords[shift_axis_ix] + shift) % L
    return qbt_ix_from_coords(temp_coords, L)


############################################
########## Build hypercubic lattices
############################################

def get_hypercube_lattice_edges(L, num_dims):
    qbt_ixs = list(range(L ** num_dims))
    edges = []
    for dim_ix in range(num_dims):
        edges += [(qbt_ix, shifted_qbt_ix(qbt_ix, +1, dim_ix, L, num_dims)) for qbt_ix in qbt_ixs]
    return qbt_ixs, edges


#########################################################
########## Get input/output qubits indexes for hypercubes
#########################################################

def get_inout_sites(L, num_dims, inout_axis=None, periodic=True):
    if periodic:
        last_ix = int(L / 2)
    else:
        last_ix = L - 1

    if inout_axis == None:
        inout_axis_ix = num_dims - 1
    else:
        inout_axis_ix = inout_axis

    indexs_vals = [list(range(L)) for _ in range(num_dims - 1)]
    indexs_vals.insert(inout_axis_ix, [0])

    input_qbts = np.array([qbt_ix_from_coords(coords, L) for coords in product(*indexs_vals)])

    indexs_vals[inout_axis_ix] = [last_ix]
    output_qbts = np.array([qbt_ix_from_coords(coords, L) for coords in product(*indexs_vals)])
    return input_qbts, output_qbts


###############################################################
########## Build probabilistic adjacency matrix with edge loss
###############################################################

def get_adj_mat_from_edges(edges, num_qbts, p_loss=0):
    amat = np.zeros((num_qbts, num_qbts)).astype(np.uint8)

    if p_loss > 0:
        alive_edges_ixs = np.where(np.random.binomial(1, 1 - p_loss, len(edges)))[0]

        for edge_ix in alive_edges_ixs:
            this_edge = edges[edge_ix]
            amat[this_edge[0], this_edge[1]] = 1
            amat[this_edge[1], this_edge[0]] = 1

    else:
        for this_edge in edges:
            amat[this_edge[0], this_edge[1]] = 1
            amat[this_edge[1], this_edge[0]] = 1

    return amat


def get_wmat_from_edges(edges, num_qbts, p_loss=0):
    amat = get_adj_mat_from_edges(edges, num_qbts, p_loss=p_loss)
    return np.block([np.identity(num_qbts).astype(np.uint8), amat])


#################################################################################################
########## Build random spherical subset of neighbouring qubits with maximum lattice distance d
#################################################################################################


def get_random_sphere_qbts_subset(amat, max_d, seed_qbt=None):
    amat_id = np.identity(amat.shape[0]).astype(np.uint8) + amat
    if seed_qbt == None:
        qbt0 = np.random.randint(amat.shape[1])
    else:
        qbt0 = seed_qbt

    arr0 = np.zeros(amat.shape[0])
    arr0[qbt0] = 1

    for _ in range(max_d):
        arr0 = amat_id @ arr0

    return np.where(arr0)[0]

#################################################################################################
########## Functions to calculate properties for a set of qubits in a stabilizer state
#################################################################################################

def wmat_stab_gens(wmat):

    """
    Function that does Gaussian elimination on a set of stabilizers (in the w matrix form) to extract the generetors
    """

    # adding the last row just to make it compatible with the gauss elim. C++ function used for loss decoding
    # TODO: change C++ function to remove this overhead
    wmat_larger = np.block([[wmat], [np.zeros((1, wmat.shape[1])).astype(np.uint8)]])

    gauss_elim_mat = loss_decoding_gausselim_fast(wmat_larger, wmat_larger.shape[1], print=False)
    #     nonzero_stabs = np.where(np.any(gauss_elim_mat, axis=1))[0]
    #     return gauss_elim_mat[nonzero_stabs]

    # removing the last row that was used to make it compatible with the gauss elim. C++ function used for loss decoding
    # TODO: change C++ function to remove this overhead
    gauss_elim_mat_nolast = gauss_elim_mat[:-1]

    return gauss_elim_mat_nolast


###############################################################################
########### Function to find stabs genetors for stab group with support only in a qubit subset
###############################################################################

def stabs_in_qubits_subset(wmat, qbts_in_subset):

    """
    Function to find stabs genetors for stab group with support only in a qubit subset
    """

    # adding the last row just to make it compatible with the gauss elim. C++ function used for loss decoding
    # TODO: change C++ function to remove this overhead
    wmat_larger = np.block([[wmat], [np.zeros((1, wmat.shape[1])).astype(np.uint8)]])

    N_qbts = int(wmat.shape[1] / 2)

    rows_in_wmat = np.block([np.array(qbts_in_subset), N_qbts + np.array(qbts_in_subset)]).astype(int)

    N_lostrow = len(rows_in_wmat)

    qbts_ixs = np.arange(2 * N_qbts).astype(int)

    temp_ixs = copy(qbts_ixs[:N_lostrow])
    lost_notmoved = rows_in_wmat[np.where(np.in1d(rows_in_wmat, temp_ixs, invert=True, assume_unique=True))]
    moved_notlost = temp_ixs[np.where(np.in1d(temp_ixs, rows_in_wmat, invert=True, assume_unique=True))]

    qbts_ixs[:N_lostrow] = rows_in_wmat
    qbts_ixs[lost_notmoved] = moved_notlost

    wmat_ordered = wmat_larger[:, qbts_ixs]

    wmat_ordered_flipped = np.flip(wmat_ordered, axis=1)

    gauss_elim_mat_flipped = loss_decoding_gausselim_fast(wmat_ordered_flipped, 2 * N_qbts - N_lostrow, print=False)
    gauss_elim_mat = np.flip(gauss_elim_mat_flipped, axis=1)

    # removing the last row that was used to make it compatible with the gauss elim. C++ function used for loss decoding
    # TODO: change C++ function to remove this overhead
    gauss_elim_mat_nolast = gauss_elim_mat[:-1]

    return gauss_elim_mat[np.where(np.logical_not(np.any(gauss_elim_mat_nolast[:, N_lostrow:], axis=1)))[0], :N_lostrow]

##############################################################################################
########### Functions to calculate the Renyi Entropy and Renyi mutual information for a subset/subsets of qubits
##############################################################################################

def get_Renyi_entropy(wmat, qbts_in_subset):
    return len(qbts_in_subset) - (stabs_in_qubits_subset(wmat, qbts_in_subset).shape[0])

def get_Renyi_mutual_info(wmat, qbts_in_subset1, qbts_in_subset2):
    return get_Renyi_entropy(wmat, qbts_in_subset1) + get_Renyi_entropy(wmat, qbts_in_subset2) - get_Renyi_entropy(wmat, np.union1d(qbts_in_subset1, qbts_in_subset2))

##############################################################################################
########### Functions to calculate the stab generators for the graph state associate to a graph
##############################################################################################
def gens_wmat_from_graph(graph):
    return get_wmat_from_edges(list(graph.edges), graph.number_of_nodes(), p_loss=0)



##############################################################################################
##############################################################################################
########### Functions to generate the stabilizers for a fusion network and move around it
##############################################################################################
##############################################################################################


############################################################################
#### Functions for cases where all qubits are fused (i.e. no static qubit)
############################################################################


def full_resource_states_wmat_fusionsonly(network_adjmat, resource_graph):
    nnodes = resource_graph.number_of_nodes()
    nsites = network_adjmat.shape[0]

    resource_wmat = gens_wmat_from_graph(resource_graph)

    #     print(resource_wmat)

    tot_num_qubits = nsites * nnodes

    wmat = np.block(
        [np.identity(tot_num_qubits).astype(np.uint8), np.zeros((tot_num_qubits, tot_num_qubits)).astype(np.uint8)])

    for site_ix in range(nsites):
        qbt_ixs = np.arange(site_ix * nnodes, (site_ix + 1) * nnodes).astype(np.uint64)
        w_ixs = np.block([qbt_ixs, qbt_ixs + tot_num_qubits])

        wmat[np.ix_(qbt_ixs, w_ixs)] = resource_wmat

    return wmat


def qbtix_from_site_and_resource_qbtix(site_ix, resource_qbtix, n_resourcegraphnodes):
    return site_ix * n_resourcegraphnodes + resource_qbtix


def site_and_resource_qbtix_from_qbtix(qbtix, n_resourcegraphnodes):
    return int(qbtix / n_resourcegraphnodes), qbtix % n_resourcegraphnodes


############################################################################
#### Functions for cases where you have both static and fused qubits
############################################################################


'''
The standard, i.e. keeping  inout_qbtix = None, is to consider the last qubit of input/output resource state 
as the static qubit.

The order of the qubits considered to build the w stabilizer generators matrix is:

[ static qubits in input sites, 
static qubits in output sites, 
fused qubits from all sites according to the site order and the qbt order within the resource state]

'''


def full_resource_states_wmat_with_inout(nsites, resource_graph_fusion, insites, outsites, inout_resource_graph,
                                         inout_qbtix=None):
    nnodes_fusion = resource_graph_fusion.number_of_nodes()
    nnodes_inout = inout_resource_graph.number_of_nodes()

    if nnodes_inout != (nnodes_fusion + 1):
        raise ValueError(
            'Number of qubits in in/out resource graphs should be the same as the fused resource states +1')

    if inout_qbtix == None:
        inout_qbt = nnodes_inout - 1
    else:
        inout_qbt = inout_qbtix

    n_insites = len(insites)
    n_outsites = len(outsites)
    n_inout_sites = n_insites + n_outsites
    n_fusedsites = nsites - n_inout_sites

    resource_wmat_fusion = gens_wmat_from_graph(resource_graph_fusion)

    resource_wmat_inout = gens_wmat_from_graph(inout_resource_graph)

    tot_num_qubits = n_fusedsites * nnodes_fusion + nnodes_inout * n_inout_sites

    wmat = np.block(
        [np.identity(tot_num_qubits).astype(np.uint8), np.zeros((tot_num_qubits, tot_num_qubits)).astype(np.uint8)])

    for site_ix in range(nsites):

        if (site_ix in insites):

            ix_in_inout = np.argmax(insites == site_ix)
            qbt_ixs = np.insert(
                np.arange(n_inout_sites + site_ix * nnodes_fusion, n_inout_sites + (site_ix + 1) * nnodes_fusion),
                inout_qbt, ix_in_inout).astype(np.uint64)
            w_ixs = np.block([qbt_ixs, qbt_ixs + tot_num_qubits])
            wmat[np.ix_(qbt_ixs, w_ixs)] = resource_wmat_inout

        elif (site_ix in outsites):

            ix_in_inout = np.argmax(outsites == site_ix)

            qbt_ixs = np.insert(
                np.arange(n_inout_sites + site_ix * nnodes_fusion, n_inout_sites + (site_ix + 1) * nnodes_fusion),
                inout_qbt, ix_in_inout + n_insites).astype(np.uint64)
            w_ixs = np.block([qbt_ixs, qbt_ixs + tot_num_qubits])
            wmat[np.ix_(qbt_ixs, w_ixs)] = resource_wmat_inout

        else:

            qbt_ixs = np.arange(n_inout_sites + site_ix * nnodes_fusion,
                                n_inout_sites + (site_ix + 1) * nnodes_fusion).astype(np.uint64)
            w_ixs = np.block([qbt_ixs, qbt_ixs + tot_num_qubits])
            wmat[np.ix_(qbt_ixs, w_ixs)] = resource_wmat_fusion

    return wmat


def qbtix_from_site_and_resource_qbtix_with_inout(site_ix, resource_qbtix, n_fusednodes, insites, outsites,
                                                  inout_qbtix=None):
    if inout_qbtix == None:
        inout_qbt = n_fusednodes
    else:
        inout_qbt = inout_qbtix

    if resource_qbtix == inout_qbt:
        if site_ix in insites:
            return site_ix
        elif site_ix in outsites:
            return site_ix + len(insites)
        else:
            raise ValueError('Tried to use an input/output qubit on a fusion-only resource state.')
            return

    else:
        return len(insites) + len(outsites) + site_ix * n_fusednodes + resource_qbtix


def site_and_resource_qbtix_from_qbtix_with_inout(qbtix, n_fusednodes, insites, outsites, inout_qbtix=None):
    if inout_qbtix == None:
        inout_qbt = n_fusednodes
    else:
        inout_qbt = inout_qbtix

    n_insites = len(insites)
    n_outsites = len(outsites)
    n_inout_sites = n_insites + n_outsites

    if qbtix < n_insites:
        return insites[qbtix], inout_qbt
    elif n_insites <= qbtix < n_inout_sites:
        return outsites[qbtix - n_insites], inout_qbt
    else:
        return int((qbtix - n_inout_sites) / n_fusednodes), (qbtix - n_inout_sites) % n_fusednodes


############################################################################
#### Functions for rules to fuse qubits in the resource states of the fusion network
############################################################################


'''
Results are specified in a dictionary as follows:
Each key represents a qubit in a resource state, 
the first element of the value is the qubit index within a resource state it is going to be fused with,
the second element of the array is the vector to move to the site whose resource state the qubit is going to fuse with.

For example, for resource graphs with fusion qubits [0, 1, 2, 3], a possible valid set of rules is:

fusion_rules = {
    0: (2, np.array([+1])),
    1: (3, np.array([+1])),
    2: (0, np.array([-1])),
    3: (1, np.array([-1]))
}

Which can also be compacted to 
fusion_rules = {
    0: (2, np.array([+1])),
    1: (3, np.array([+1]))
}


Consistency properties to have uniform fusion rules:

- If qbt1 is to be fused with qbt2 with translation vec(x), then,
  if qbt2 is present in the list, it has to be fused with qbts1 and with translation -vec(x).
- The set union of the keys and the qubits in the values has to be the full fused qubits list 
  (i.e. all fusion qubits have to be fused).
- Vec(x) can't be zero (i.e. no fusions allowed in the same resource state
                        ...just start with the resulting one for more efficiency).

  These conditions imply
  1) No qubit_ix can be fused with the same qubit ix (otherwise vec(x) is 0, which is not allowed).
  2) The number of fused qubits is always even.
'''


def check_fusion_rules_consistency_and_compact(fusion_rules, fused_qubits_list):
    in_fused_list = list(fusion_rules.keys())
    out_fused_list = [x[0] for x in fusion_rules.values()]

    compacted_fusion_rules = {}

    if not np.array_equal(np.sort(np.union1d(in_fused_list, out_fused_list)), np.sort(fused_qubits_list)):
        raise ValueError('Inconsistency in fusion rules: not all fusion qubits are fused')
        return

    for in_fused in fusion_rules:
        (out_fuse, in_vec) = fusion_rules[in_fused]

        if np.array_equal(in_vec, np.zeros(in_vec.shape)):
            raise ValueError('Inconsistency in fusion rules: translation vector cant be zero.')
            return

        if out_fuse == in_fused:
            raise ValueError(
                'Inconsistency in fusion rules: a qubit from a resource state cannot be fused with the one from another resource state with the same index.')
            return

        if out_fuse in fusion_rules:
            (new_in_fuse, new_vec) = fusion_rules[out_fuse]

            if new_in_fuse != in_fused:
                raise ValueError('Inconsistency in fusion rules: a qubit is assigned to multiple fusions.')
                return

            if not np.array_equal(in_vec, -new_vec):
                raise ValueError('Inconsistency in fusion rules: a fusion is assigned to multiple translations.')
                return

        if (in_fused not in compacted_fusion_rules) and (out_fuse not in compacted_fusion_rules):
            compacted_fusion_rules[in_fused] = (out_fuse, in_vec)

    return compacted_fusion_rules


####################################################################
######## Function to get the stabilizer generators (as a w matrix) of a fusion network
######## made out of resource states in a hypercube.
####################################################################


def hypercube_fusion_network_with_inout(L, num_dims, resource_graph_fusion, inout_resource_graph, fusion_rules,
                                        inout_qbtix=None):
    sites, lattice_edges = get_hypercube_lattice_edges(L, num_dims)
    nsites = len(sites)
    network_amat = get_adj_mat_from_edges(lattice_edges, nsites)

    in_sites, out_sites = get_inout_sites(L, num_dims)

    nnodes_fusion = resource_graph_fusion.number_of_nodes()
    nnodes_inout = inout_resource_graph.number_of_nodes()

    n_insites = len(in_sites)
    n_outsites = len(out_sites)
    n_inout_sites = n_insites + n_outsites
    n_fusedsites = nsites - n_inout_sites

    if nnodes_inout != (nnodes_fusion + 1):
        raise ValueError(
            'Number of qubits in in/out resource graphs should be the same as the fused resource states +1')

    if inout_qbtix == None:
        inout_qbt = nnodes_inout - 1
    else:
        inout_qbt = inout_qbtix

    full_resource_wmat = full_resource_states_wmat_with_inout(nsites, resource_graph_fusion, in_sites, out_sites,
                                                              inout_resource_graph)

    fused_resource_qbt_ixs = list(inout_resource_graph.nodes)
    fused_resource_qbt_ixs.remove(inout_qbt)

    compacted_fusion_rules = check_fusion_rules_consistency_and_compact(fusion_rules, fused_resource_qbt_ixs)

    fusion_network = np.zeros((int(nsites * nnodes_fusion / 2), 2)).astype(np.uint64)

    num_qbts_in_rules = len(compacted_fusion_rules)

    for site_ix in sites:
        site_coords = coords_from_qbt_ix(site_ix, L, num_dims)

        for local_ix, res_qbt_ix in enumerate(compacted_fusion_rules):
            (other_res_qbt_ix, site_shift_vect) = fusion_rules[res_qbt_ix]

            new_site_coords = np.mod(site_coords + site_shift_vect, L)

            new_site_ix = qbt_ix_from_coords(new_site_coords, L)

            fuse_in_ix = qbtix_from_site_and_resource_qbtix_with_inout(site_ix, res_qbt_ix, nnodes_fusion, in_sites,
                                                                       out_sites, inout_qbtix=inout_qbtix)
            fuse_out_ix = qbtix_from_site_and_resource_qbtix_with_inout(new_site_ix, other_res_qbt_ix, nnodes_fusion,
                                                                        in_sites, out_sites, inout_qbtix=inout_qbtix)

            fusion_network[site_ix * num_qbts_in_rules + local_ix] = np.array([fuse_in_ix, fuse_out_ix]).astype(
                np.uint64)

    return full_resource_wmat, fusion_network, in_sites, out_sites



def get_wmat_on_inout(wmat, in_sites, out_sites):
    num_all_qbts = int(wmat.shape[1] / 2)
    num_insites = len(in_sites)
    num_outsites = len(out_sites)

    inout_ixs_array = np.block(
        [np.arange(num_insites + num_outsites), np.arange(num_insites + num_outsites) + num_all_qbts])

    wmat_inout_gens = wmat_stab_gens(wmat[:, inout_ixs_array])
    return wmat_inout_gens[np.where(np.any(wmat_inout_gens, axis=1))]



################################################################################
##### Functions to update stabilizers upon fusion measurements
################################################################################


def update_fusion_xx_success(fused_xx_pairs_list, wmat):
    """
    This assumes for simplicity that wmat is an identity matrix (i.e. always do xx fusions
    before zz fusions!!!).
    TODO: this could be made more general (see zz update function), but would probably also slow it.
    """

    removed_stabs_ixs = np.ones(len(fused_xx_pairs_list)) * (wmat.shape[0] + 1)
    for fusion_ix, fused_pair in enumerate(fused_xx_pairs_list):
        w_mat_xx_pair = wmat[:, fused_pair]

        xx_fused_stab_ixs = np.argmax(w_mat_xx_pair, axis=0)
        wmat[xx_fused_stab_ixs[0]] = (wmat[xx_fused_stab_ixs[0]] + wmat[xx_fused_stab_ixs[1]]) % 2

        removed_stabs_ixs[fusion_ix] = xx_fused_stab_ixs[1]

    kept_ixs = [i for i in range(wmat.shape[0]) if i not in removed_stabs_ixs]
    #     print(kept_ixs)

    wmat = wmat[kept_ixs]

    return wmat


def update_fusion_zz_success(fused_zz_pairs_list, wmat):
    removed_stabs_ixs = np.ones(len(fused_zz_pairs_list)) * (wmat.shape[0] + 1)

    for fusion_ix, fused_pair in enumerate(fused_zz_pairs_list):

        zzfused_w_ixs = fused_pair + int(wmat.shape[1] / 2)
        #         print()
        #         print('Starting new fusion!\n\n')
        #         print(fused_pair)
        #         print(zzfused_w_ixs)
        #         print()

        #         print(wmat[:, zzfused_w_ixs])

        cols_diff = wmat[:, [zzfused_w_ixs[0]]] - wmat[:, [zzfused_w_ixs[1]]]
        rows_in1_not2 = np.where(cols_diff == np.array([1], dtype=np.uint8))[0]
        rows_in2_not1 = np.where(cols_diff == np.array([-1], dtype=np.uint8))[0]
        rows_both12 = np.where(wmat[:, [zzfused_w_ixs[0]]] + wmat[:, [zzfused_w_ixs[1]]] == 2)[0]
        rows_all12 = np.block([rows_both12, rows_in1_not2, rows_in2_not1])

        #         print(rows_in1_not2)
        #         print(rows_in2_not1)
        #         print(rows_both12)
        #         print(rows_all12)
        #         print()

        wmat_zz = wmat[rows_all12]
        wmat_zz_1not2 = wmat[rows_in1_not2]
        wmat_zz_2not1 = wmat[rows_in2_not1]

        #         print(wmat_zz)
        #         print()
        #         print(wmat_zz_1not2)
        #         print()
        #         print(wmat_zz_2not1)
        #         print()

        num_1not2 = len(rows_in1_not2)
        num_2not1 = len(rows_in2_not1)
        num_both12 = len(rows_both12)

        #         print('num_both12', num_both12)
        #         print('num_1not2', num_1not2)
        #         print('num_2not1', num_2not1)
        #         print((num_both12 + (num_1not2 * num_2not1), wmat.shape[1]))

        wmat_zz_fused = np.zeros(
            (num_both12 + max(num_1not2 * num_2not1, num_1not2 + num_2not1), wmat.shape[1])).astype(np.uint8)

        #         print('\nInitialised wmat_zz_fused')
        #         print(wmat_zz_fused)

        for ix_both12 in range(num_both12):
            wmat_zz_fused[ix_both12] = wmat[rows_both12[ix_both12]]

        for ix_1not2 in range(num_1not2):
            for ix_2not1 in range(num_2not1):
                #                 print()
                #                 print('num_1not2  num_2not1', num_1not2, num_2not1)
                #                 print('ix_1not2   ix_1not2  num_both12', ix_1not2, ix_2not1, num_both12)
                #                 print()
                wmat_zz_fused[num_both12 + ix_1not2 * num_2not1 + ix_2not1] = (wmat_zz_1not2[ix_1not2] + wmat_zz_2not1[
                    ix_2not1]) % 2

        wmat_zz_fused = wmat_stab_gens(wmat_zz_fused)

        #         print('\nAfter-fusion and Gaussian elimination wmat_zz_fused')
        #         print(wmat_zz_fused)

        zeroed_stab_ix = rows_all12[np.where(np.logical_not(np.any(wmat_zz_fused[:len(rows_all12)], axis=1)))[0]]

        wmat[rows_all12] = wmat_zz_fused[:len(rows_all12)]

        #         print(wmat)
        #         print()
        if len(zeroed_stab_ix) > 0:
            removed_stabs_ixs[fusion_ix] = zeroed_stab_ix[0]

    kept_ixs = [i for i in range(wmat.shape[0]) if i not in removed_stabs_ixs]

    wmat = wmat[kept_ixs]

    return wmat


def update_fusion_lost(lost_xx_pairs_list, lost_zz_pairs_list, wmat):
    # def update_fusion_lost(qbts_in_subset, wmat):

    #     adding the last row just to make it compatible with the gauss elim. C++ function used for loss decoding
    #     TODO: change C++ function to remove this overhead
    wmat_larger = np.block([[wmat], [np.zeros((1, wmat.shape[1])).astype(np.uint8)]])

    xx_lost_cols = lost_xx_pairs_list.reshape((1, lost_xx_pairs_list.shape[0] * lost_xx_pairs_list.shape[1]))[0]
    zz_lost_cols = lost_zz_pairs_list.reshape((1, lost_zz_pairs_list.shape[0] * lost_zz_pairs_list.shape[1]))[0]

    N_qbts = int(wmat.shape[1] / 2)

    cols_in_wmat = np.block([xx_lost_cols, N_qbts + zz_lost_cols]).astype(int)

    N_lostrow = len(cols_in_wmat)

    qbts_ixs = np.arange(2 * N_qbts).astype(int)

    temp_ixs = copy(qbts_ixs[:N_lostrow])

    lost_notmoved = cols_in_wmat[np.where(np.in1d(cols_in_wmat, temp_ixs, invert=True, assume_unique=True))]
    moved_notlost = temp_ixs[np.where(np.in1d(temp_ixs, cols_in_wmat, invert=True, assume_unique=True))]

    qbts_ixs[:N_lostrow] = cols_in_wmat
    qbts_ixs[lost_notmoved] = moved_notlost

    reversed_qbts_ixs = np.zeros(qbts_ixs.shape).astype(int)
    for ix, el in enumerate(qbts_ixs):
        reversed_qbts_ixs[el] = ix

    wmat_ordered = wmat_larger[:, qbts_ixs]

    gauss_elim_mat = loss_decoding_gausselim_fast(wmat_ordered, N_lostrow, print=False)

    # removing the last row that was used to make it compatible with the gauss elim. C++ function used for loss decoding
    # TODO: change C++ function to remove this overhead
    gauss_elim_mat_nolast = gauss_elim_mat[:-1]

    gauss_elim_mat_nolast = gauss_elim_mat_nolast[
        np.where(np.logical_not(np.any(gauss_elim_mat_nolast[:, :N_lostrow], axis=1)))]

    gauss_elim_mat_nolast = gauss_elim_mat_nolast[:, reversed_qbts_ixs]

    return gauss_elim_mat_nolast


################################################################################
##### Functions to get the mutual information between input and output static qubits after the network has been fused.
################################################################################


def mutual_info_for_probabilistic_fusion_network(wmat_fusion_network, fused_qubits, p_fusion_fail, in_sites, out_sites):
    wmat = copy(wmat_fusion_network)

    zz_fusion_eras_errors = np.random.binomial(1, p_fusion_fail, len(fused_qubits))
    zz_fusion_successful = fused_qubits[np.where(np.logical_not(zz_fusion_eras_errors))]
    zz_fusion_failed = fused_qubits[np.where(zz_fusion_eras_errors)]

    xx_fusion_eras_errors = np.random.binomial(1, p_fusion_fail, len(fused_qubits))
    xx_fusion_successful = fused_qubits[np.where(np.logical_not(xx_fusion_eras_errors))]
    xx_fusion_failed = fused_qubits[np.where(xx_fusion_eras_errors)]

    new_full_resource_wmat = update_fusion_xx_success(xx_fusion_successful, wmat)
    new_full_resource_wmat = update_fusion_zz_success(zz_fusion_successful, new_full_resource_wmat)
    new_full_resource_wmat = update_fusion_lost(xx_fusion_failed, zz_fusion_failed, new_full_resource_wmat)

    ixs_in = np.arange(len(in_sites))
    ixs_out = np.arange(len(out_sites)) + len(in_sites)

    inout_wmat = get_wmat_on_inout(new_full_resource_wmat, ixs_in, ixs_out)

    return get_Renyi_mutual_info(inout_wmat, ixs_in, ixs_out)
