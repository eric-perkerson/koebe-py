"""Script to run the edge chaining function and export new regions"""
import sys
import os
from pathlib import Path
import numpy as np

import region as reg
import chain


def test_chain_edges():
    edges = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0]]
        ]
    )
    test = chain.chain_edges(edges)
    expected_result = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ]
    )
    assert np.all(test == expected_result)


def main(full_file_stem):
    """Main method"""
    region_path = Path(full_file_stem)
    region = reg.Region.read_poly(region_path.with_suffix('.poly'))

    components = region.get_components()

    upper_edges = chain.read_edges_file(full_file_stem + '.upper_edges.txt')
    lower_edges = chain.read_edges_file(full_file_stem + '.lower_edges.txt')

    upper_chain = chain.chain_edges(upper_edges)
    lower_chains = chain.chain_edges_multi_connected_components(lower_edges)

    print(f'len(upper_chain) = {len(upper_chain)}')
    print(f'len(lower_chains) = {len(lower_chains[0])}')
    print(f'len(lower_chains) = {len(lower_chains[1])}')

    # Construct region from the outer boundary and upper_edges
    upper_region = reg.Region([components[0], upper_chain])
    with open(full_file_stem + '.upper_region.poly', 'w', encoding='utf-8') as file:
        upper_region.write(file)

    # Construct regions from inner boundaries and lower_edges
    lower_region_1 = reg.Region([components[1], lower_chains[0]])
    lower_region_2 = reg.Region([components[2], lower_chains[1]])
    with open(full_file_stem + '.lower_region_1.poly', 'w', encoding='utf-8') as file:
        lower_region_1.write(file)
    with open(full_file_stem + '.lower_region_2.poly', 'w', encoding='utf-8') as file:
        lower_region_2.write(file)

    print(upper_region)


if __name__ == '__main__':
    REGION_DIR = '/Users/eric/Code/combinatorial-topology/regions'
    if len(sys.argv) > 1:
        region_name = sys.argv[1]
        # region_name = 'africa_round_upper_region'
        # region_name = 'africa_round'
        full_file_stem = os.path.join(REGION_DIR, region_name, region_name)
        print(f'Writing to {full_file_stem}')
        main(full_file_stem)
    else:
        raise Exception('Need to specify file stem')
