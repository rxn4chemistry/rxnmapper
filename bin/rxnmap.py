#!/usr/bin/env python3

import argparse
import sys
import time

from rxnmapper import RXNMapper

rxn_mapper = RXNMapper()

def get_confidence(mapping):
    return mapping['confidence']

def get_mapping(mapping):
    return mapping['mapped_rxn']

if __name__ == '__main__':
    before = time.time()
    # CLI options parsing
    parser = argparse.ArgumentParser(description = "Atom-atom mapping of reaction SMILES")
    parser.add_argument("-i", metavar = "input.rsmi", dest = "input_fn",
                        help = "reactions input file")
    parser.add_argument("-o", metavar = "output.rsmi", dest = "output_fn",
                        help = "mapped reactions output file")
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
        tokens = line.split()
        rsmi = tokens[0]
        # print("line: %s" % line)
        # print("rsmi: %s" % rsmi)
        mapped = rxn_mapper.get_attention_guided_atom_maps([rsmi])
        mapping = get_mapping(mapped[0])
        print('%s' % mapping, file=output)
        count += 1
    after = time.time()
    dt = after - before
    print("read %d reactions at %.2f Hz" %
          (count, count / dt), file=sys.stderr)
    output.close()
