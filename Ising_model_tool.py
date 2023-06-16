import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random

def get_bound_EA(index1, index2, bonds):
    bond = frozenset({tuple(index1),tuple(index2)})
    if bond in bonds:
        return bonds[bond]
    else:
        bonds[bond] = np.random.standard_normal()
        return bonds[bond]

def get_bound_RF(index1, index2, bonds):
    bond = frozenset({tuple(index1),tuple(index2)})
    if bond in bonds:
        return bonds[bond]
    else:
        bonds[bond] = np.abs(np.random.standard_normal())
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
    for j in range(len(indices)):
        if (indices[j] == 0):
            neighbor_index.append(increase_entry_by_one(indices, j))
            indice_copy = indices.copy()
            indice_copy[j] = L-1
            neighbor_index.append(indice_copy)
        elif (indices[j] == L-1):
            neighbor_index.append(decrease_entry_by_one(indices, j))
            indice_copy = indices.copy()
            indice_copy[j] = 0
            neighbor_index.append(indice_copy)
        else:
            neighbor_index.append(increase_entry_by_one(indices, j))
            neighbor_index.append(decrease_entry_by_one(indices, j))
    return neighbor_index

def get_energy_EA(spin, spin_index, neighbor_index, S, bonds):
    energy = 0
    for neighbor in neighbor_index:
        bond = get_bound_EA(spin_index, neighbor, bonds)
        energy = energy + bond*spin*S[tuple(neighbor)]
    return energy

def get_energy_RF(spin, spin_index, neighbor_index, S, bonds):
    energy = 0
    for neighbor in neighbor_index:
        bond = get_bound_RF(spin_index, neighbor, bonds)
        energy = energy + bond*spin*S[tuple(neighbor)]
    return energy

def overlap(S1,S2,N):
    return np.sum(S1*S2)/N

def sweep_EA(S,L,N,bonds):
    sweep = 0
    while sweep < N:
        indices = [np.random.choice(dim) for dim in S.shape]
        spin = S[tuple(indices)]
        neighbor_index = get_neighbor(indices,L)
        beforeE = get_energy_EA(spin, indices, neighbor_index, S, bonds)
        afterE = get_energy_EA(-spin, indices, neighbor_index, S, bonds)
        deltaE = afterE - beforeE
        if deltaE > 0:
            S[tuple(indices)] = -spin
        sweep = sweep+1

def is_active_EA(index, S, L, bonds):
    spin = S[tuple(index)]
    neighbor_index = get_neighbor(np.asarray(index),L)
    beforeE = get_energy_EA(spin, index, neighbor_index, S, bonds)
    afterE = get_energy_EA(spin, index, neighbor_index, S, bonds)
    deltaE = afterE - beforeE
    return deltaE > 0

def get_active_EA(S,L,bonds):
    it = np.nditer(S,flags = ['multi_index'])
    active_indices = []
    while not it.finished:
        index = it.multi_index
        if is_active_EA(index, S, L, bonds):
            active_indices.append(index)
        it.iternext()
    return active_indices

def kineticMonteCarlo_EA(S,L,active_list,bonds):
    l = len(active_list)
    if l == 0:
        return 0
    t = 1/l
    index = random.choice(active_list)
    spin = S[tuple(index)]
    neighbor_index = get_neighbor(np.asarray(index),L)
    beforeE = get_energy_EA(spin, index, neighbor_index, S, bonds)
    afterE = get_energy_EA(-spin, index, neighbor_index, S, bonds)
    deltaE = afterE - beforeE
    if deltaE > 0:
        S[tuple(index)] = -spin
        active_list.remove(tuple(index))
        for nspin in neighbor_index:
            if is_active_EA(nspin,S,L):
                if not (tuple(nspin) in active_list):
                    active_list.append(tuple(nspin))
            else:
                if (tuple(nspin) in active_list):
                    active_list.remove(tuple(nspin))
    return t

def sweep_RF(S,L,N,bonds):
    sweep = 0
    while sweep < N:
        indices = [np.random.choice(dim) for dim in S.shape]
        spin = S[tuple(indices)]
        neighbor_index = get_neighbor(indices,L)
        beforeE = get_energy_RF(spin, indices, neighbor_index, S, bonds)
        afterE = get_energy_RF(-spin, indices, neighbor_index, S, bonds)
        deltaE = afterE - beforeE
        if deltaE > 0:
            S[tuple(indices)] = -spin
        sweep = sweep+1

def is_active_RF(index, S, L, bonds):
    spin = S[tuple(index)]
    neighbor_index = get_neighbor(np.asarray(index),L)
    beforeE = get_energy_RF(spin, index, neighbor_index, S, bonds)
    afterE = get_energy_RF(spin, index, neighbor_index, S, bonds)
    deltaE = afterE - beforeE
    return deltaE > 0

def get_active_RF(S,L,bonds):
    it = np.nditer(S,flags = ['multi_index'])
    active_indices = []
    while not it.finished:
        index = it.multi_index
        if is_active_RF(index, S, L, bonds):
            active_indices.append(index)
        it.iternext()
    return active_indices

def kineticMonteCarlo_RF(S,L,active_list,bonds):
    l = len(active_list)
    if l == 0:
        return 0
    t = 1/l
    index = random.choice(active_list)
    spin = S[tuple(index)]
    neighbor_index = get_neighbor(np.asarray(index),L)
    beforeE = get_energy_RF(spin, index, neighbor_index, S, bonds)
    afterE = get_energy_RF(-spin, index, neighbor_index, S, bonds)
    deltaE = afterE - beforeE
    if deltaE > 0:
        S[tuple(index)] = -spin
        active_list.remove(tuple(index))
        for nspin in neighbor_index:
            if is_active_RF(nspin,S,L):
                if not (tuple(nspin) in active_list):
                    active_list.append(tuple(nspin))
            else:
                if (tuple(nspin) in active_list):
                    active_list.remove(tuple(nspin))
    return t