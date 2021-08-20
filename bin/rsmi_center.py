#!/usr/bin/env python3
#
# Copyright (C) 2021, Francois Berenger
# Tsuda laboratory, Tokyo University,
# 5-1-5 Kashiwa-no-ha, Kashiwa-shi, Chiba-ken, 277-8561, Japan.
#
# Compute Chemical Distance (CD) and reaction centers from a reaction SMILES

import argparse
import math
import numpy
import rdkit
import sys
import time

from rdkit import Chem

# the string is a mixture, in SMILES format
def molecules_of_string(dotted_smiles):
    res = []
    for s in dotted_smiles.split("."):
        mol = Chem.MolFromSmiles(s)
        if type(mol) == rdkit.Chem.rdchem.Mol:
            res.append(mol)
        else:
            print('Not a molecule: %s' % s, file = sys.stderr)
    return res

def element_count(mols):
    res = {}
    for mol in mols:
        for a in mol.GetAtoms():
            anum = a.GetAtomicNum()
            try:
                prev = res[anum]
                res[anum] = prev + 1
            except KeyError:
                res[anum] = 1
    return res

def is_balanced(reactants, products):
    return (element_count(reactants) == element_count(products))

# atom int identifier from a mapped reaction SMILES is gotten by:
# <a:rdkit_atom>.GetAtomMapNum()

# get all bonds between mapped atoms
# list the bonds as: (lower_mapped_atom, bond_order, higher_mapped_atom)
def list_mapped_atom_bonds(mapped_mol):
    res = []
    for a in mapped_mol.GetAtoms():
        a_map_num = a.GetAtomMapNum()
        if a_map_num > 0:
            a_i = a.GetIdx()
            for b in a.GetNeighbors():
                b_map_num = b.GetAtomMapNum()
                if b_map_num > a_map_num:
                    b_i = b.GetIdx()
                    bond = mapped_mol.GetBondBetweenAtoms(a_i, b_i)
                    bond_order = bond.GetBondTypeAsDouble()
                    res.append((a_map_num, bond_order, b_map_num))
    return res

# create a map: (mapped_i, mapped_j) -> bond_order
# with mapped_i < mapped_j
def list_mapped_bonds(mapped_mols):
    res = {}
    for m in mapped_mols:
        for (a_i, bo, b_i) in list_mapped_atom_bonds(m):
            res[(a_i, b_i)] = bo
    return res

# chemical distance: a sum of abs(delta(bond_order))
# only between mapped atoms (Warning: Hs are ignored by most AAM algorithms)
def compare_bonds(mapped_reactants, mapped_products):
    res = 0.0
    react_centers = set()
    # rm bonds which did not change
    # for bonds whose order has changed, res += abs(delta(bond_order))
    # for bonds which apppeared, res += bond_order
    # for bonds which disappeared, res += bond_order
    left = list_mapped_bonds(mapped_reactants)
    right = list_mapped_bonds(mapped_products)
    for k1 in left:
        bo1 = left[k1]
        (i, j) = k1
        assert(bo1 > 0.0)
        try:
            # bond sets intersection
            bo2 = right[k1]
            if bo1 != bo2:
                res += math.fabs(bo1 - bo2)
                react_centers.add(i)
                react_centers.add(j)
        except KeyError:
            # bond only in left
            res += bo1
            react_centers.add(i)
            react_centers.add(j)
    for k2 in right:
        bo2 = right[k2]
        (i, j) = k2
        try:
            # bond sets intersection; already processed above
            bo1 = left[k2]
        except KeyError:
            # bond only in right
            res += bo2
            react_centers.add(i)
            react_centers.add(j)
    # Python sets are not sorted; hence the sorted(react_centers)...
    return (res, sorted(react_centers))

if __name__ == '__main__':
    before = time.time()
    # CLI options parsing
    parser = argparse.ArgumentParser(description = "process reaction SMILES")
    parser.add_argument("-i", metavar = "input.rsmi", dest = "input_fn",
                        help = "reactions input file")
    parser.add_argument("-o", metavar = "output.rsmi", dest = "output_fn",
                        help = "reactions output file \
                        (with CD and reaction centers)")
    # parse CLI
    if len(sys.argv) == 1:
        # user has no clue of what to do -> usage
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    input_fn = args.input_fn
    output = open(args.output_fn, 'w')
    count = 0
    # read reactions ---------------------------------------------------------
    for raw_line in open(input_fn, 'r').readlines():
        line = raw_line.strip()
        # print("line: %s" % line)
        react_reag_prods = line.split(">")
        # assert(len(react_prods) == 2)
        (reactants, _reagents, products) = tuple(react_reag_prods)
        # print("reactants: %s" % reactants)
        # print("products: %s" % products)
        react_mols = molecules_of_string(reactants)
        prod_mols = molecules_of_string(products)
        if not is_balanced(react_mols, prod_mols):
            print("IMBALANCED\t%s" % line, file = sys.stderr)
        cd, centers = compare_bonds(react_mols, prod_mols)
        print("%s\t%.1f\t%s" % (line, cd, centers), file = output)
        count += 1
    after = time.time()
    dt = after - before
    print("read %d reactions at %.2f mol/s" %
          (count, count / dt), file=sys.stderr)
    output.close()
