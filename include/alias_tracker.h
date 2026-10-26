#ifndef AMIGO_ALIAS_TRACKER_H
#define AMIGO_ALIAS_TRACKER_H

#include <stdexcept>
#include <vector>

namespace amigo {

template <typename I>
class AliasTracker {
 public:
  AliasTracker(I size) : parent(size), rank(size, 0) {
    for (I i = 0; i < size; ++i) {
      parent[i] = i;
    }
  }

  size_t size() const { return parent.size(); }

  I find(I var) {
    if (parent[var] != var) {
      parent[var] = find(parent[var]);  // Path compression
    }
    return parent[var];
  }

  void alias(std::vector<I>& vec1, std::vector<I>& vec2) {
    if (vec1.size() != vec2.size()) {
      throw std::invalid_argument("Linked arrays must have the same length");
    }
    for (size_t i = 0; i < vec1.size(); i++) {
      I var1 = vec1[i];
      I var2 = vec2[i];
      I root1 = find(var1);
      I root2 = find(var2);

      // A pair already in one group leaves the remaining pairs to link
      if (root1 == root2) {
        continue;
      }

      // Union by rank
      if (rank[root1] < rank[root2]) {
        parent[root1] = root2;
      } else {
        parent[root2] = root1;
        if (rank[root1] == rank[root2]) {
          rank[root1]++;
        }
      }
    }
  }

  I assign_group_vars(I* ptr) {
    I array_size = size();
    std::vector<I> root_to_id(array_size, -1);
    I counter = 0;

    for (I i = 0; i < array_size; i++) {
      I root = find(i);
      if (root_to_id[root] == -1) {
        root_to_id[root] = counter++;
      }
      ptr[i] = root_to_id[root];
    }

    return counter;
  }

 private:
  std::vector<I> parent;
  std::vector<I> rank;
};

}  // namespace amigo

#endif  // AMIGO_ALIAS_TRACKER_H