r"""cluster.py -- the cluster channel: how much of a wallet's volume stays inside its community.

Wash trading moves no beneficial ownership, so a wash group's volume is traded inside the group.
That is the defining property, and no execution choice can hide it: v8's interleaving, pipeline
and slow rate change *when* the group trades, not *with whom*. Detection v5 measured it with the
labels (internal share 0.60 against 0.31 under volume-proportional mixing, active). This module
finds candidate groups without labels and scores them.

1. **Communities.** Louvain (networkx, modularity, weight = the pro-rata counterparty volume V of
   v1's pairing, fixed seed) partitions the wallets. Resolution 1 is networkx's default and the
   main setting; resolution 2 (smaller communities) is a sensitivity check.

2. **Wallet score.** For wallet i in community C:

       a_i = V[i, C] / V[i, :]                      share of i's volume inside C
       e_i = (vol(C) - vol_i) / (tot - vol_i)       the same share if i's counterparties were drawn
                                                    in proportion to volume (mixing)
       s_i = (a_i - e_i) / max(1 - e_i, a_i)

   s_i is the smaller of two normalisations of the excess a_i - e_i:
     * (a - e) / (1 - e), the share of the possible excess that is realised. It fails for a
       community that holds most of the market: if a wash group carries half of all volume and
       trades with itself, the honest market is a closed community too, with a = 0.86, e = 0.53,
       (a - e) / (1 - e) = 0.70 (v8 ``honest`` sweep, block-executed rounds, active);
     * 1 - e / a, the share of i's in-community volume beyond mixing (a / e is the lift). It fails
       for small communities, where a few trades give a large lift.
   The minimum needs both: at the floor 0.5, half of the possible excess and twice the mixing
   share. s_i = 0 where it is undefined (singleton community, one community covering every wallet).

3. **Pair score.** min(s_i, s_j) clipped at 0 when i and j are in the same community, 0 when they
   are not, NaN for self-pairs (as in every other channel). The minimum is Sethi's pair rule. It
   matters here: a pool wallet that placed one camouflage order sits in its counterparty's
   community with a high score, but its honest counterparty does not, so their pair is not flagged.

4. **Where the channel applies.** The score tests "this group trades with itself more than a mixing
   market predicts". That needs a mixing honest market in which the group is a minority. When a
   wash group carries most of the volume (thin, thick: 56-88%), the honest wallets form the small
   closed block and look exactly like a group; no graph statistic can tell which block is the
   wash group. The channel is therefore only used in backgrounds whose control runs mix: mean
   modularity of the control partitions below ``MAX_CONTROL_MODULARITY`` (active backgrounds
   0.15-0.19; thin and thick 0.39-0.47, where honest trading alone already forms strong
   communities). Elsewhere the cluster scores are set to 0 and v6 equals v4.

Screening history (seeds 0-9 of the v8 wash cells): scoring whole communities flagged about 40% of
honest volume in wash runs (leaf wallets inflate honest communities' internal share); wallet scores
with the pair minimum fixed that. (a - e) / (1 - e) alone flagged about half of the honest volume in
the block-executed cells (see 2.); iterative peeling of the most anomalous community did not fix it.

Unlike v4's channels, the cluster score is not propagated over the graph: the community already
is the network context.
"""
import networkx as nx
import numpy as np

MAIN_RESOLUTION = 1.0
RESOLUTIONS = (1.0, 2.0)
SEED = 0
MAX_CONTROL_MODULARITY = 0.3


def channel_name(resolution) -> str:
    return "cluster" if resolution == MAIN_RESOLUTION else f"cluster_r{resolution:g}"


def communities(V: np.ndarray, resolution=MAIN_RESOLUTION, seed=SEED) -> np.ndarray:
    """Louvain community label of every wallet (0..k-1)."""
    n = V.shape[0]
    lab = np.zeros(n, dtype=int)
    if n == 0:
        return lab
    G = nx.Graph()
    G.add_nodes_from(range(n))
    iu = np.argwhere(np.triu(V, 1) > 0)
    G.add_weighted_edges_from((int(i), int(j), float(V[i, j])) for i, j in iu)
    for k, c in enumerate(nx.community.louvain_communities(G, weight="weight",
                                                            resolution=resolution, seed=seed)):
        lab[list(c)] = k
    return lab


def wallet_scores(V: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """s_i for every wallet (see the module docstring)."""
    vol = V.sum(axis=1)
    tot = vol.sum()
    s = np.zeros(len(vol))
    for k in np.unique(lab):
        idx = np.where(lab == k)[0]
        if len(idx) < 2:
            continue
        vC = vol[idx].sum()
        a = V[np.ix_(idx, idx)].sum(axis=1) / vol[idx]
        e = (vC - vol[idx]) / (tot - vol[idx])
        with np.errstate(divide="ignore", invalid="ignore"):
            si = (a - e) / np.maximum(1.0 - e, a)
        s[idx] = np.where(np.isfinite(si), si, 0.0)
    return s


def pair_scores(lab: np.ndarray, s: np.ndarray, bi: np.ndarray, si: np.ndarray,
                in_graph: np.ndarray) -> np.ndarray:
    """Cluster score of every pair; ``bi`` / ``si`` are graph positions (ignored where not
    ``in_graph``, i.e. for self-pairs, which get NaN)."""
    out = np.full(len(bi), np.nan)
    b, k = bi[in_graph].astype(int), si[in_graph].astype(int)
    same = lab[b] == lab[k]
    out[in_graph] = np.where(same, np.clip(np.minimum(s[b], s[k]), 0.0, None), 0.0)
    return out


def modularity(V: np.ndarray, lab: np.ndarray) -> float:
    """Newman modularity of the partition on the weighted graph."""
    tot = V.sum()
    if tot <= 0:
        return float("nan")
    vol = V.sum(axis=1)
    q = 0.0
    for k in np.unique(lab):
        idx = np.where(lab == k)[0]
        q += V[np.ix_(idx, idx)].sum() / tot - (vol[idx].sum() / tot) ** 2
    return float(q)
