import pyop3
import time


if __name__ == "__main__":
    t0 = time.time()

    S0 = 40
    K = 45
    T = 3
    r = 0.07
    sigma = 0.3
    N = 4*3

    tree = pyop3.binomial_tree(S0, r, T, N, sigma=sigma, tree_type='CRR')
    option = pyop3.american_option(underlying_asset=tree, strike=K)

    Prime = option.put()
    # Prime = option.call()
    t1 = time.time()

    print(Prime)
    print(t1 - t0)
    
    pyop3.tree_planter.show_tree(tree=option.asset_tree)
