from collections import defaultdict

import numpy as np

from CAMASim.function.module.distance import (
    crossbar_innerproduct_pairwise,
    euclidean_pairwise,
    hamming_pairwise,
    innerproduct_pairwise,
    manhattan_pairwise,
    rangequery_pairwise,
)
from CAMASim.function.module.merge import exact_merge, knn_merge, threshold_merge, topk_merge
from CAMASim.function.module.sensing import (
    get_array_best_results,
    get_array_best_results_sensing,
    get_array_exact_results,
    get_array_threshold_results,
    get_array_topk_results,
)


class CAMSearch:
    def __init__(self, query_config, array_config):
        """
        Initializes the CAMSearch class with configuration settings.

        Args:
            query_config (dict): Configuration settings for query operations.
            array_config (dict): Configuration settings for the CAM array.
        """
        self.query_config  = query_config
        self.array_config  = array_config
        self.metric        = self.define_distance_metrics()

        self.searchScheme    = query_config['searchScheme']   # "exact" | "knn" | "topk"
        self.searchParameter = query_config['parameter']      # k for knn/topk, threshold otherwise
        self.sensing         = array_config['sensing']        # "exact" | "best" | "threshold" | "topk"
        self.sensinglimit    = array_config.get('sensingLimit', 0)

        # hw noise params forwarded from query_config (used by crossbar_ip metric)
        self.hw = query_config.get('hw', {})

        # Optional PerfEval top-k cost accounting (Part 2). Off unless
        # query_config['perfeval'] is provided, e.g.:
        #   {"comparator_cost": {...}, "model": "linear", "num_comparators": 1}
        # When on, the top-k hardware cost is accumulated at the sensing and
        # merge stages during a real search -- the in-loop counterpart to
        # perfeval.py, hooked exactly where partial results are pooled/picked.
        self.perfeval = query_config.get('perfeval', None)
        self._pe_topk_comparisons = None
        if self.perfeval is not None:
            from CAMASim.performance.topk_cost import topk_comparisons as _tc
            self._pe_topk_comparisons = _tc
        self._reset_perfeval()

    def _reset_perfeval(self):
        """Zero the top-k cost accumulators (called at the start of each search)."""
        self._pe_sensing_cmp = 0
        self._pe_merge_cmp = 0

    def get_perfeval_report(self):
        """Finalize the top-k cost accumulated during the last search().

        Returns None if PerfEval accounting was not enabled. Otherwise a dict
        with the comparison counts -- split into the size-dependent per-array
        'sensing' selection and the cross-array 'merge' re-rank -- and the
        aggregated hardware cost using the comparator_cost from
        query_config['perfeval'].
        """
        if self.perfeval is None:
            return None
        import math as _math
        cc = self.perfeval["comparator_cost"]
        a = float(cc["area_per_comparator_m2"])
        l = float(cc["latency_per_comparison_s"])
        e = float(cc["energy_per_comparison_j"])
        num_comp = int(self.perfeval.get("num_comparators", 1))
        total_cmp = self._pe_sensing_cmp + self._pe_merge_cmp
        return {
            "sensing_comparisons": self._pe_sensing_cmp,
            "merge_comparisons": self._pe_merge_cmp,
            "topk_comparisons_total": total_cmp,
            "topk_energy_j": e * total_cmp,
            "topk_latency_s": (_math.ceil(total_cmp / num_comp) * l) if total_cmp else 0.0,
            "topk_area_m2": a * num_comp,
            "num_comparators": num_comp,
        }

    def define_search_area(self, numRowCAMs, numColCAMs):
        self.numRowCAMs = numRowCAMs
        self.numColCAMs = numColCAMs

    def search(self, cam_data, query_data):
        """
        Perform a search operation across all CAM arrays.

        Args:
            cam_data   : Data stored in the CAM arrays.
            query_data : Query data.

        Returns:
            results (list): List of search results per query.
        """
        if self.perfeval is not None:
            self._reset_perfeval()
        matchInd     = defaultdict(list)
        matchIndDist = defaultdict(list)
        rowSize      = self.array_config['row']

        # 1. Search in each array
        for i in range(self.numRowCAMs):
            for j in range(self.numColCAMs):
                indices, distances = self.array_search(cam_data[i, j], query_data[:, j])
                for m in range(query_data.shape[0]):
                    curr_indices = [row + i * rowSize for row in indices[m]]
                    matchInd[m]     += curr_indices
                    matchIndDist[m] += list(distances[m]) if hasattr(distances[m], '__iter__') else [distances[m]]

        # 2. Merge results
        results = []
        for m in range(query_data.shape[0]):
            result = self.merge_indices(matchInd[m], matchIndDist[m])
            results.append(result)
        return results

    def array_search(self, data, query):
        distance_matrix = self.array_distance(data, query)
        indices, distances = self.array_sensing(distance_matrix)
        return indices, distances

    def array_distance(self, data, query):
        return self.metric(data, query)

    def array_sensing(self, distance_matrix):
        """
        Sensing within a single array.

        For 'topk' sensing the score_matrix contains INNER PRODUCTS
        (higher = better).  All other modes expect a distance matrix
        (lower = better).
        """
        if self.sensing == 'exact':
            indices, distances = get_array_exact_results(distance_matrix)
        elif self.sensing == 'best':
            if self.sensinglimit != 0:
                indices, distances = get_array_best_results_sensing(distance_matrix, self.sensinglimit)
            else:
                indices, distances = get_array_best_results(distance_matrix)
        elif self.sensing == 'threshold':
            indices, distances = get_array_threshold_results(distance_matrix, self.searchParameter)
        elif self.sensing == 'topk':
            # distance_matrix here is actually a score matrix (inner products)
            indices, distances = get_array_topk_results(distance_matrix, self.searchParameter)
            if self._pe_topk_comparisons is not None:
                # Local per-array top-k selects k out of this array's candidates
                # for every query. Summed over all arrays this is ~k*n_docs --
                # the size-dependent cost that grows with the database.
                sm = np.asarray(distance_matrix)
                n_cand = sm.shape[1] if sm.ndim > 1 else sm.shape[0]
                n_q = sm.shape[0] if sm.ndim > 1 else 1
                self._pe_sensing_cmp += self._pe_topk_comparisons(
                    int(n_cand), int(self.searchParameter)) * int(n_q)
        else:
            raise NotImplementedError(f"Unknown sensing mode: {self.sensing}")
        return indices, distances

    def merge_indices(self, matchInd, matchIndDist):
        """
        Merge per-array results into a single ranked list.

        'topk' searchScheme re-ranks by score across all row-arrays.
        All other schemes are unchanged from the original CAMASim logic.
        """
        if self.searchScheme == 'exact':
            result = exact_merge(matchInd, self.numRowCAMs, self.numColCAMs)
        elif self.searchScheme == 'knn':
            result = knn_merge(matchInd, self.numRowCAMs, self.numColCAMs, self.searchParameter)
        elif self.searchScheme == 'threshold':
            result = threshold_merge(matchInd, self.numRowCAMs, self.numColCAMs)
        elif self.searchScheme == 'topk':
            result = topk_merge(matchInd, matchIndDist, self.searchParameter)
            if self._pe_topk_comparisons is not None:
                # Cross-array merge re-ranks the pooled candidates for this query
                # -- the exact spot the draft describes hooking the top-k in.
                n_pool = len(matchInd)
                if n_pool > 0:
                    self._pe_merge_cmp += self._pe_topk_comparisons(
                        int(n_pool), int(self.searchParameter))
        else:
            print("Please choose a valid search scheme.")
            raise NotImplementedError
        return result

    def define_distance_metrics(self):
        metric = self.query_config['distance']

        if not callable(metric):
            metrics = {
                "euclidean":    euclidean_pairwise,
                "manhattan":    manhattan_pairwise,
                "hamming":      hamming_pairwise,
                "innerproduct": innerproduct_pairwise,
                "rangequery":   rangequery_pairwise,
            }

            if metric == "crossbar_ip":
                hw = self.query_config.get('hw', {})
                return lambda data, query: crossbar_innerproduct_pairwise(data, query, hw)

            metric = metrics.get(metric, euclidean_pairwise)

        if self.query_config['searchScheme'] == 'exact':
            metric = metrics.get(metric, hamming_pairwise)

        if self.array_config['cell'] == "ACAM":
            metric = metrics.get(metric, rangequery_pairwise)

        return metric