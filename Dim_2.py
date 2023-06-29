import numpy as np
import scipy as sp
import multiprocessing
import os
from mpi4py import MPI

def get_bound(index1, index2, bonds):
    bond = frozenset({tuple(index1),tuple(index2)})
    if bond in bonds:
        return bonds[bond]
    else:
        bonds[bond] = np.random.standard_normal()
        return bonds[bond]

def increase_entry_by_one(A, j):
    A_modified = A.copy()
    A_modified[j] += 1
    return A_modified
def decrease_entry_by_one(A, j):
    A_modified = A.copy()
    A_modified[j] -= 1
    return A_modified

def get_neighbor(indices,L):
    neighbor_index = []
    for j in range(len(L)):
        if (indices[j] == 0):
            neighbor_index.append(increase_entry_by_one(indices, j))
            indice_copy = indices.copy()
            indice_copy[j] = L[j]-1
            neighbor_index.append(indice_copy)
        elif (indices[j] == L[j]-1):
            neighbor_index.append(decrease_entry_by_one(indices, j))
            indice_copy = indices.copy()
            indice_copy[j] = 0
            neighbor_index.append(indice_copy)
        else:
            neighbor_index.append(increase_entry_by_one(indices, j))
            neighbor_index.append(decrease_entry_by_one(indices, j))
    return neighbor_index

def get_energy(spin, spin_index, neighbor_index, S):
    energy = 0
    for neighbor in neighbor_index:
        bond = get_bound(spin_index, neighbor, bonds)
        energy = energy + bond*spin*S[tuple(neighbor)]
    return energy

def overlap(S1,S2,N):
    return np.sum(S1*S2)/N

def sweep(S,L,N):
    sweep = 0
    while sweep < N:
        indices = [np.random.choice(dim) for dim in L]
        spin = S[tuple(indices)]
        neighbor_index = get_neighbor(indices,L)
        beforeE = get_energy(spin, indices, neighbor_index, S)
        afterE = get_energy(-spin, indices, neighbor_index, S)
        deltaE = afterE - beforeE
        if deltaE > 0:
            S[tuple(indices)] = -spin
        sweep = sweep+1

def is_active(index, S, L):
    spin = S[tuple(index)]
    neighbor_index = get_neighbor(np.asarray(index),L)
    beforeE = get_energy(spin, index, neighbor_index, S)
    afterE = get_energy(-spin, index, neighbor_index, S)
    deltaE = afterE - beforeE
    return deltaE > 0

def get_active(S,L):
    it = np.nditer(S,flags = ['multi_index'])
    active_indices = []
    while not it.finished:
        index = it.multi_index
        if is_active(index, S, L):
            active_indices.append(index)
        it.iternext()
    return active_indices

import random

def kineticMonteCarlo(S,L,active_list):
    l = len(active_list)
    if l == 0:
        return 0
    t = 1/l
    index = random.choice(active_list)
    spin = S[tuple(index)]
    neighbor_index = get_neighbor(np.asarray(index),L)
    beforeE = get_energy(spin, index, neighbor_index, S)
    afterE = get_energy(-spin, index, neighbor_index, S)
    deltaE = afterE - beforeE
    if deltaE > 0:
        S[tuple(index)] = -spin
        active_list.remove(tuple(index))
        for nspin in neighbor_index:
            if is_active(nspin,S,L):
                if not (tuple(nspin) in active_list):
                    active_list.append(tuple(nspin))
            else:
                if (tuple(nspin) in active_list):
                    active_list.remove(tuple(nspin))
    return t

def montecarlmethod(S,L,N):
    survival1 = 0
    survival2 = 0
    S2 = S.copy()
    S1 = S.copy()
    bonds = dict()
    while len(get_active(S1,L)) != 0:
        sweep(S1,L,N)
        survival1 = survival1 + 1
        if survival1 == 10:
            break

    S1_active = get_active(S1,L)

    while True:
        k = kineticMonteCarlo(S1,L,S1_active)
        if k == 0:
            break
        survival1 = survival1 + k

    while len(get_active(S2,L)) != 0:
        sweep(S2,L,N)
        survival2 = survival2 + 1
        if survival2 == 10:
            break

    S2_active = get_active(S2,L)

    while True:
        k = kineticMonteCarlo(S2,L,S2_active)
        if k == 0:
            break
        survival2 = survival2 + k

    return [overlap(S1,S2,N),survival1,survival2] 

bonds = dict()



if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    # Get the rank and size of the MPI communicator
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_samples = size

    # Generate the 100x100 arrays for each sample
    fixed_arg1 = np.array([100, 100])
    if rank == 0:
        array_args = [np.random.choice([-1, 1], size=tuple(fixed_arg1)) for _ in range(num_samples)]
    else:
        array_args = None
    # Create the fixed arguments
    fixed_arg2 = np.prod(fixed_arg1)

    # Distribute the workload across processes using scatter
    local_array_args = comm.bcast(array_args, root=0)

    sample_index = rank
    array_arg = local_array_args[sample_index]
    
    
    result = montecarlmethod(array_arg, fixed_arg1, fixed_arg2)

    # Gather the results from all processes
    all_results = comm.gather(result, root=0)

    if rank == 0:
                # Concatenate the results from all processes
        combined_results = all_results

                # Print the final results
        print(combined_results)

            # Finalize MPI
    MPI.Finalize()



'''

if __name__ == '__main__':
    num_samples = 96

    num_processes = multiprocessing.cpu_count()  # Number of available CPU cores
        # Generate the 100x100 arrays for each sample
    fixed_arg1 = np.array([100, 100])
    array_args = [np.random.choice([-1, 1], size=tuple(fixed_arg1)) for _ in range(num_samples)]
        # Create the fixed arguments
    fixed_arg2 = np.prod(fixed_arg1)

        # Combine the array arguments with the fixed arguments
    args_list = [(array_arg, fixed_arg1, fixed_arg2) for array_arg in array_args]

        # Create a multiprocessing pool
    pool = multiprocessing.Pool(processes = num_processes)

        # Perform the Monte Carlo simulations in parallel
    results = pool.map(montecarlmethod, args_list)

        # Close the multiprocessing pool
    print('End')
    pool.close()
    pool.join()

    print(results)
    overlaps = [sublist[0] for sublist in results]
    survival1 = [sublist[1] for sublist in results]
    survival2 = [sublist[2] for sublist in results]
    q_2 = np.mean(overlaps)
    survival1_2 = np.mean(survival1)
    survival2_2 = np.mean(survival2)
    print(q_2)
    print(survival1_2)
    print(survival2_2)'''