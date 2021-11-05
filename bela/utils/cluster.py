import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import logging

class Grinch(object):

    def __init__(
        self,
        points,
        rotate_cap=100,
        graft_cap=100,
        sim='dot',
        norm='l2',
        active_leaf_limit=None,
        pruning_strategy='least_recent',
        pruning_threshold=None,
    ):

        self.points = points
        self.num_points = points.shape[0]
        self.dim = points.shape[1]
        self.max_nodes = 3 * self.num_points  # 3 because of lazy grafting implementation.
        self.rotate_cap = rotate_cap
        self.graft_cap = graft_cap
        self.sim = sim
        self.norm = norm

        self.active_leaf_limit = active_leaf_limit if active_leaf_limit is not None else self.num_points
        self.pruning_strategy = pruning_strategy
        self.pruning_threshold = pruning_threshold

        self.csim = self.get_csim(sim)
        self.compute_centroid = self.get_compute_centroid(norm)
        self.select_for_pruning = self.get_select_for_pruning(pruning_strategy)

        # Bookkeeping
        self.centroids = np.zeros((self.max_nodes, self.dim), dtype=np.float32)
        self.sums = np.zeros((self.max_nodes, self.dim), dtype=np.float32)

        self.ancs = [[] for _ in range(self.max_nodes)]
        self.sibs = None
        self.children = [[] for _ in range(self.max_nodes)]
        self.descendants = [[] for _ in range(self.max_nodes)]
        self.scores = -np.inf * np.ones(self.max_nodes, dtype=np.float32)
        self.needs_update_model = np.zeros(self.max_nodes, dtype=np.bool_)
        self.needs_update_desc = np.zeros(self.max_nodes, dtype=np.bool_)
        self.parents = -1 * np.ones(self.max_nodes, dtype=np.int32)
        self.next_node_id = self.num_points
        self.num_descendants = -1 * np.ones(self.max_nodes, dtype=np.float32)

        # Added for memory bounding.
        self.active_leaves = np.zeros(self.max_nodes, dtype=np.bool_)
        self.first_used = np.zeros(self.max_nodes, dtype=np.int32)
        self.current_step = 0

    # Similarity Functions
    def get_csim(self, sim):
        METHODS ={
            'dot': self.csim_dot,
            'l2': self.csim_l2,
            'sql2': self.csim_sql2,
        }
        assert sim in METHODS, f"Sim. fct. '{sim}' not in {list(METHODS.keys())}"
        return METHODS[sim]

    @staticmethod
    def csim_dot(x, y):
        sims = np.matmul(x, y.transpose(1, 0))
        return sims

    @staticmethod
    def csim_l2(x, y):
        dists = cdist(x, y)
        return 1.0 / (1 + dists)

    @staticmethod
    def csim_sql2(x, y):
        dists = cdist(x, y, 'sqeuclidean')
        return 1.0 / ( 1 + dists)

    # Centroid Functions
    def get_compute_centroid(self, norm):
        METHODS = {
            'l2': self.compute_centroid_l2_norm,
            'l_inf': self.compute_centroid_l_inf_norm,
            'none': self.compute_centroid_no_norm,
        }
        assert norm in METHODS, f"Norm '{norm}' not in {list(METHODS.keys())}"
        return METHODS[norm]

    def compute_centroid_l2_norm(self, i):
        self.centroids[i] *= 0
        self.centroids[i] += self.sums[i]
        self.centroids[i] /= self.num_descendants[i]
        if type(i) is np.array:
            norms = np.linalg.norm(self.centroids[i], axis=1, keepdims=True)
            norms[norms==0.0] = 1.0
        else:
            norms = np.linalg.norm(self.centroids[i])
            norms = norms if norms > 0 else 1.0
        self.centroids[i] /= norms

    def compute_centroid_l_inf_norm(self, i):
        self.centroids[i] *= 0
        self.centroids[i] += self.sums[i]
        self.centroids[i] /= self.num_descendants[i]
        self.centroids[i] /= np.linalg.norm(self.centroids[i], np.inf)

    def compute_centroid_no_norm(self, i):
        self.centroids[i] *= 0
        self.centroids[i] += self.sums[i]
        self.centroids[i] /= self.num_descendants[i]

    # Pruning functions
    def select_most_similar(self, candidate_ids):
        """Selects node with most similar children to prune."""
        scores = self.get_score_batch(candidate_ids)
        argmax = np.argmax(scores)
        return candidate_ids[argmax]

    def select_least_recent(self, candidate_ids):
        """Selects least recently seen node to prune."""
        recent = self.first_used[candidate_ids]
        argmin = np.argmin(recent)
        return candidate_ids[argmin]

    def select_combined(self, candidate_ids):
        """Selects most similar node if similarity above a given threshold."""
        scores = self.get_score_batch(candidate_ids)
        argmax = np.argmax(scores)
        if scores[argmax] > self.pruning_threshold:
            return candidate_ids[argmax]
        return self.select_least_recent(candidate_ids)

    def get_select_for_pruning(self, pruning_strategy):
        METHODS = {
            'similarity': self.select_most_similar,
            'least_recent': self.select_least_recent,
            'combined': self.select_combined,
        }
        assert pruning_strategy in METHODS, \
            f"Pruning strategy '{pruning_strategy}' not in {list(METHODS.keys())}"
        if pruning_strategy == 'combined':
            assert self.pruning_threshold is not None, 'Combined pruning needs threshold.'
        return METHODS[pruning_strategy]

    # Main GRINCH Operations
    def build_dendrogram(self):
        """Builds the dendrogram."""
        for i in tqdm(range(self.num_points), 'grinch_build_dendrogram'):
            self.insert(i)

    def insert(self, i):
        """Inserts point i into the dendrogram."""
        logging.debug('[insert] insert(%s)', i)

        if self.current_step == 0:
            self.add_pt(i)
        else:
            # Compute nearest neighbors before adding point (to avoid being near to self).
            i_vec = np.expand_dims(self.points[i], 0)
            dists, nns = self.cknn(i_vec)

            # Add the point.
            self.add_pt(i)

            # Identify sibling (looking for rotations from nearest neighboring leaf node).
            sib = self.find_rotate(i, nns[0])

            # Create new parent node above sibling and make current node and sibling its children.
            parent = self.node_from_nodes(sib, i)
            self.make_sibling(sib, i, parent)

            # Mark parents for update.
            curr_update = parent
            while curr_update != -1:
                self.updated_from_children(curr_update)
                curr_update = self.parents[curr_update]

            # Perform graft operation.
            self.graft(parent)

        # If we've exceeded the memory threshold, then we need to collapse 2 nodes into their
        # parent (collapsing 1 node is not sufficient since adding a point adds the point and its
        # new parent).
        if self.active_leaves.sum() > self.active_leaf_limit:
            self.prune()

        self.current_step += 1

    def add_pt(self, i):
        """Initializes relevant metadata for newly added point."""
        self.sums[i] = self.points[i]
        self.num_descendants[i] = 1
        self.descendants[i].append(i)
        self.compute_centroid(i)
        self.first_used[i] = self.current_step
        self.active_leaves[i] = True

    def cknn(self, i_vec, offlimits1=None, offlimits2=None):
        """Computes the nearest neighbors of a point."""

        # Changed. Initialize similairities to a vector of all -inf, then only compute similarities
        # for active nodes (to reduce runtime early on). While before just the list of points was
        # used this now considers all nodes.
        sims = np.full(
            shape=(self.max_nodes,),
            fill_value= -float("Inf"),
            dtype=np.float32,
        )
        point_vecs = self.centroids[self.active_leaves]
        sims[self.active_leaves] = self.csim(i_vec, point_vecs).squeeze(0)

        if offlimits1 is not None:
            sims[offlimits1] = -float("Inf")
        if offlimits2 is not None:
            sims[offlimits2] = -float("Inf")

        indices = np.argmax(sims)
        distances = sims[indices]
        indices = indices[distances != -np.Inf]
        distances = distances[distances != -np.Inf]

        return distances, indices

    def find_rotate(self, gnode, sib):
        """Given a newly added node `gnode` and its nearest neighbor `sib` determine whether an
        ancestor of `sib` is suited for a rotate operation."""
        logging.debug('[rotate] find_rotate(%s, %s)', gnode, sib)
        curr = sib
        score = self.e_score(gnode, sib)
        curr_parent = self.parents[curr]
        curr_parent_score = -np.Inf if curr_parent == -1 else self.get_score(curr_parent)
        while curr_parent != -1 \
                and score < curr_parent_score \
                and self.num_descendants[curr_parent] < self.rotate_cap:
            logging.debug('[rotate] curr %s curr_parent %s gnode %s score %s curr_parent_score %s', curr, curr_parent,
                          gnode, score, curr_parent_score)
            curr = curr_parent
            curr_parent = self.parents[curr]
            curr_parent_score = -np.Inf if curr_parent == -1 else self.get_score(curr_parent)
            score = self.e_score(gnode, sib)
        logging.debug('[rotate] find_rotate(%s, %s) = %s', gnode, sib, curr)
        return curr

    def node_from_nodes(self, n1, n2):
        logging.debug('[node_from_nodes] creating new node from nodes %s and %s', n1, n2)
        new_node_id = self.next_node_id
        logging.debug('[node_from_nodes] new node is %s', new_node_id)
        assert self.next_node_id < self.max_nodes
        self.next_node_id += 1
        self.needs_update_model[new_node_id] = True
        self.needs_update_desc[new_node_id] = True
        self.num_descendants[new_node_id] = self.num_descendants[n1] + self.num_descendants[n2]
        if self.num_descendants[new_node_id] == 0:
            raise RuntimeError('Zero descendants')
        self.first_used[new_node_id] = self.current_step
        return new_node_id

    def make_sibling(self, node, new_sib, parent):
        sib_parent = self.parents[new_sib]
        logging.debug('[make_sibling] make_sibling(node=%s, new_sib=%s, parent=%s) sib_parent=%s', node, new_sib, parent, sib_parent)

        # The following routine only pertains to grafts.
        if sib_parent != -1:
            logging.debug('[make_sibling] this message should only appear for grafts.')
            sib_gp = self.parents[sib_parent]
            old_sib = self.get_sibling(new_sib)
            self.parents[old_sib] = sib_gp
            if sib_gp != -1:
                self.remove_child(sib_gp, sib_parent)
                self.add_child(sib_gp, old_sib)
            self.clear_children(sib_parent)
            self.parents[sib_parent] = -2 # Code for deletion
        else:
            assert self.active_leaves[new_sib]

        # Transfer parent from existing node to new parent.
        grandparent = self.parents[node]
        self.parents[parent] = grandparent
        logging.debug('[make_sibling] grandparent=%i', grandparent)

        # If the grandparent is not null, then replace its former child (now grandchild) with the
        # newly created parent.
        if grandparent != -1:
            self.remove_child(grandparent, node)
            self.add_child(grandparent, parent)

        # Add children to new parent & vice versa.
        self.add_child(parent, node)
        self.add_child(parent, new_sib)
        self.parents[node] = parent
        self.parents[new_sib] = parent

    def graft(self, gnode):
        logging.debug('[graft] graft(%s)', gnode)
        curr = gnode

        # Do not graft onto descendants...
        offlimits1 = self.get_descendants(curr)

        # ...or sibling (if it is a leaf)
        offlimits2 = []
        if self.parents[curr] != -1:
            sibling = self.get_sibling(curr)
            if self.active_leaves[sibling]:
                offlimits2 = [sibling]

        # logging.debug('[graft] len(offlimits1)=%s len(offlimits2)=%s', len(offlimits1), len(offlimits2))
        # logging.debug('[graft] offlimits1 %s offlimits2 %s', str(offlimits1), str(offlimits2))

        # Find updates
        self.update(curr)
        curr_v = np.expand_dims(self.centroids[curr], 0)

        # Do search
        _, nns = self.cknn(curr_v, offlimits1, offlimits2)
        if len(nns) == 0:
            logging.debug('[graft] No nearest neighbors after nns....')
            return

        oneNN = nns[0]
        logging.debug('[graft] Nearest neighbor is %s', oneNN)
        lca, this2anc, other2anc = self.lca_and_ancestors(gnode, oneNN)
        logging.debug('[graft] lca %s len(this2anc) %s len(other2anc) %s', lca, len(this2anc), len(other2anc))
        if this2anc and other2anc:
            # M by N
            M = len(this2anc)
            N = len(other2anc)
            score_if_grafted = self.e_score_batch(this2anc, other2anc)
            assert score_if_grafted.shape[0] == M
            assert score_if_grafted.shape[1] == N
            # 1 by N
            nn_parent_score = np.expand_dims(self.get_score_batch(self.parents[other2anc]),0)
            assert nn_parent_score.shape[0] == 1
            assert nn_parent_score.shape[1] == N
            # M by 1
            curr_parent_score = np.expand_dims(self.get_score_batch(self.parents[this2anc]), 1)
            assert curr_parent_score.shape[0] == M
            assert curr_parent_score.shape[1] == 1

            not_i_like_you = score_if_grafted <= curr_parent_score
            not_you_like_me = score_if_grafted <= nn_parent_score
            assert not_i_like_you.shape[0] == M
            assert not_i_like_you.shape[1] == N
            assert not_you_like_me.shape[0] == M
            assert not_you_like_me.shape[1] == N

            graft_condition = not_i_like_you | not_you_like_me
            num_meeting_condition = graft_condition.sum()
            total_candidate_grafts = max(1.0, len(this2anc) * len(other2anc))

            score_if_grafted[graft_condition] = 0
            argmax = np.argmax(score_if_grafted)
            argmax_row = int(argmax / score_if_grafted.shape[1])
            argmax_col = argmax % score_if_grafted.shape[1]
            best_1 = this2anc[argmax_row]
            best_2 = other2anc[argmax_col]
            if not not_i_like_you[argmax_row, argmax_col] and not not_you_like_me[argmax_row, argmax_col]:
                bestPair2gp = self.parents[self.parents[best_2]]
                parent = self.node_from_nodes(best_1, best_2)
                self.make_sibling(best_1, best_2, parent)
                logging.debug('[graft] node %s grafts node %s, scores %s > max(%s, %s)' % (best_1, best_2,
                                                                                           score_if_grafted[argmax_row,
                                                                                                            argmax_col],
                                                                                           curr_parent_score[argmax_row,0],
                                                                                           nn_parent_score[0,argmax_col]))
                for start in [bestPair2gp, self.parents[curr]]:
                    curr_update = start
                    while curr_update != -1:
                        self.updated_from_children(curr_update)
                        curr_update = self.parents[curr_update]
            else:
                logging.debug('[graft] There was no case where we wanted to graft.')

    def prune(self):
        # Nodes w/ two leaves for children are candidates for collapse.
        parent_ids, active_leaf_counts = np.unique(self.parents[self.active_leaves],
                                                   return_counts=True)
        candidate_ids = parent_ids[active_leaf_counts==2]
        pruned_id = self.select_for_pruning(candidate_ids)
        logging.debug('[prune] Pruning children of %s', pruned_id)

        # If not is not up-to-date, then make sure it is updated.
        if self.needs_update_model[pruned_id]:
            self.single_update(pruned_id)

        # Make parent active, pruned leaves inactive, and update descendants of parents.
        self.active_leaves[pruned_id] = True
        self.num_descendants[pruned_id] = 1
        self.descendants[pruned_id] = [pruned_id]
        for child_id in self.children[pruned_id]:
            self.active_leaves[child_id] = False
            self.num_descendants[child_id] = -1

        # Update parents
        curr = self.parents[pruned_id]
        while curr != -1:
            self.updated_from_children(curr)
            curr = self.parents[curr]

    def update_desc(self, i, use_tqdm=False):
        needs_update = []
        to_check = [i]
        # This is a top-down traversal from i.
        while to_check:
            curr = to_check.pop(0)
            if self.needs_update_desc[curr]:
                needs_update.append(curr)
                for c in self.children[curr]:
                    to_check.append(c)
        # For each node on the way down if it needs to be updated...
        if use_tqdm:
            for j in tqdm(range(len(needs_update)-1,-1,-1)):
                self.single_update_desc(needs_update[j])
        else:
            for j in range(len(needs_update)-1,-1,-1):
                self.single_update_desc(needs_update[j])

    def single_update_desc(self, i):
        # ...update its children
        logging.debug('[update] updating node %s', i)
        assert self.needs_update_desc[i]
        self.descendants[i].clear()
        if not self.active_leaves[i]:
            '[update] is not a leaf.'
            kids = self.children[i]
            self.descendants[i].extend(self.descendants[kids[0]])
            if len(kids) > 1:
                self.descendants[i].extend(self.descendants[kids[1]])
        else:
            '[update] is not a leaf.'
            self.descendants[i] = [i]
        self.needs_update_desc[i] = False

    def get_descendants(self, i):
        if self.needs_update_desc[i]:
            logging.debug('[get_descendants] Updating because of get_descendants!')
            self.update_desc(i)
        return self.descendants[i]

    def lca_and_ancestors(self, i, j):
        if i == j:
            return (i, [], [])
        if self.parents[i] == -1:
            logging.debug('lca_and_ancestors i = root %s', i)
            return (i, [], [])
        curr_node = j
        thisAnclist = self.get_ancs_with_self(i)
        thisAnc = dict([(nid,idx) for idx, nid in enumerate(thisAnclist)])
        other2lca = []
        while curr_node not in thisAnc:
            other2lca.append(curr_node)
            curr_node = self.parents[curr_node]
        this2lca = thisAnclist[:thisAnc[curr_node]]
        return (curr_node, [x for x in this2lca if self.num_descendants[x] < self.graft_cap],
                [x for x in other2lca if self.num_descendants[x] < self.graft_cap])

    def updated_from_children(self, i):
        self.num_descendants[i] = self.num_descendants[self.children[i][0]] + \
                                  self.num_descendants[self.children[i][1]]
        if self.num_descendants[i] <= 0:
            raise RuntimeError('Houston we have a problem')
        self.scores[i] = -float('inf')
        self.needs_update_model[i] = True
        self.needs_update_desc[i] = True

    def update(self, i, use_tqdm=False):
        needs_update = []
        to_check = [i]
        while to_check:
            curr = to_check.pop(0)
            if self.needs_update_model[curr]:
                needs_update.append(curr)
                for c in self.children[curr]:
                    to_check.append(c)
        if use_tqdm:
            for j in tqdm(range(len(needs_update)-1,-1,-1)):
                self.single_update(needs_update[j])
        else:
            for j in range(len(needs_update)-1,-1,-1):
                self.single_update(needs_update[j])
        return needs_update

    def single_update(self, i):
        logging.debug('[update] updating node %s', i)
        assert self.needs_update_model[i]
        kids = self.children[i]
        c1 = np.expand_dims(self.sums[kids[0]], 0)
        c2 = np.expand_dims(self.sums[kids[1]], 0)
        self.num_descendants[i] = self.num_descendants[kids[0]] + self.num_descendants[kids[1]]
        if self.num_descendants[i] <= 0:
            raise RuntimeError('Houston we have a problem')
        self.sums[i] = c1 + c2
        self.compute_centroid(i)
        self.needs_update_model[i] = False

    def e_score_batch(self, i, j):
        if np.any(self.needs_update_model[i]):
            for ii in i:
                if self.needs_update_model[ii]:
                    self.update(ii)
        if np.any(self.needs_update_model[j]):
            for jj in j:
                if self.needs_update_model[jj]:
                    self.update(jj)

        i_vec = self.centroids[i]
        j_vec = self.centroids[j]

        # sims = np.matmul(i_vec, j_vec.transpose(1, 0))
        sims = self.csim(i_vec, j_vec)
        return sims

    def e_score(self, i, j):
        """Updates nodes and recomputes similarity score."""
        if self.needs_update_model[i]:
            self.update(i)

        if self.needs_update_model[j]:
            self.update(j)

        i_vec = np.expand_dims(self.centroids[i], 0)
        j_vec = np.expand_dims(self.centroids[j], 0)
        sims = self.csim(i_vec, j_vec)

        return sims[0][0]

    def get_score_batch(self, i):
        # todo vectorize
        if not np.all(np.isfinite(self.scores[i])):
            for ii in i:
                if not np.isfinite(self.scores[ii]):
                    kids = self.children[ii]
                    res = self.e_score(kids[0], kids[1])
                    self.scores[ii] = res
        return self.scores[i]


    def get_score(self, i):
        """Get the linkage score at node with index i."""
        # ?! Getter should not mutate :(
        if not np.all(np.isfinite(self.scores[i])):
            kids = self.children[i]
            res = self.e_score(kids[0], kids[1])
            self.scores[i] = res
        return self.scores[i]

    def get_sibling(self, i):
        p = self.parents[i]
        return [x for x in self.children[p] if x !=i][0]

    def get_ancs_with_self(self, i):
        needs_anc = [i]
        # walk up until we find someone who has known ancestors
        curr = self.parents[i]
        while curr != -1:
            needs_anc.append(curr)
            curr = self.parents[curr]
        return needs_anc

    def add_child(self, p, c):
        self.children[p].append(c)

    def remove_child(self, p, c):
        assert c in self.children[p], 'trying to remove c=%s from p=%s with kids=%s' % (c, p, str(self.children[p]))
        self.children[p].remove(c)

    def clear_children(self, i):
        self.children[i].clear()

    def get_cluster_elements(self, node):
        """Gets the points that fall into a cluster (even if they've been pruned)."""
        frontier = [node]
        out = []
        while frontier:
            n = frontier.pop(0)
            if len(self.children[n]) == 0:
                out.append(n)
            else:
                frontier.extend(self.children[n])
        return out

    def write_tree(self, filename, lbls):
        # NOTE(rloganiv): This appears to be parent-based and iterates over the entire node list.
        # This is a good sign for "pruning", so long as we only erase information from the child
        # list.
        logging.info('writing tree to file %s', filename)
        with open(filename, 'w') as fin:
            for i in tqdm(range(self.num_points), desc='write file'):
                fin.write('%s\t%s\t%s\n' % (i, self.parents[i], lbls[i]))
            for j in range(self.num_points, self.next_node_id):
                if self.parents[j] != -2:
                    fin.write('%s\t%s\tNone\n' % (j, self.parents[j]))
            # ?!
            r = self.root()
            fin.write('-1\tNone\tNone\n' % r)

    def root(self):
        r = 0
        while self.parents[r] != -1:
            r = self.parents[r]
        return r

    def flat_clustering(self, threshold):
        frontier = [self.root()]
        clusters = []
        while frontier:
            n = frontier.pop(0)
            if len(self.children[n]) != 0 and self.get_score(n) < threshold:
                frontier.extend(self.children[n])
            else:
                clusters.append(n)
        assignments = -1*np.ones(self.num_points, np.int32)
        for c_idx, c in enumerate(clusters):
            for d in self.get_cluster_elements(c):
                assignments[d] = c_idx
        return assignments