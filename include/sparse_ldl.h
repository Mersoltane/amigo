#ifndef AMIGO_SPARSE_LDL_H
#define AMIGO_SPARSE_LDL_H

#include "blas_interface.h"
#include "csr_matrix.h"

namespace amigo {

template <typename T>
class SparseLDL {
 public:
  SparseLDL(std::shared_ptr<CSRMat<T>> mat, double ustab = 0.1,
            double pivot_growth = 2.0)
      : mat(mat), ustab(ustab), pivot_growth(pivot_growth) {
    // Get the non-zero pattern
    int nrows;
    const int *rowp, *cols;
    mat->get_data(&nrows, nullptr, nullptr, &rowp, &cols, nullptr);

    // Perform the symbolic anallysis based on the input pattern
    symbolic_analysis(nrows, rowp, cols);
  }
  ~SparseLDL() {
    delete[] snode_size;
    delete[] var_to_snode;
    delete[] snode_to_var;
    delete[] num_children;
    delete[] contrib_ptr;
    delete[] contrib_rows;
  }

  /**
   * @brief Perform a LDL^{T} factorization of the matrix
   *
   * @return int The return flag
   */
  int factor() {
    // Get the non-zero pattern
    int nrows, ncols, nnz;
    const int *rowp, *cols;
    const T* data;
    mat->get_data(&nrows, &ncols, &nnz, &rowp, &cols, &data);

    // Perform the numerical factorization
    return factor_numeric(nrows, rowp, cols, data);
  }

  /**
   * @brief Compute the solution of the system of equations
   *
   * L * D * L^{T} * y = x
   *
   * where y <- x. The solution vector overwrites the right-hand-side.
   *
   * @param xvec
   */
  void solve(Vector<T>* xvec) {
    T* x = xvec->get_array();
    T* temp = nullptr;

    int ns = 0;
    for (int ks = 0; ks < num_snodes; ks++) {
      // Get the pointers to the factor data
      int num_pivots, num_delayed;
      const int* pivots = nullptr;
      const int* delayed = nullptr;
      const T *L11, *L21;
      fact.get_factor(ks, &num_pivots, &pivots, &num_delayed, &delayed, &L11,
                      &L21);

      // Extract the variables from x
      int jj = 0;
      for (int j = 0; j < num_pivots; j++, jj++) {
        temp[jj] = x[pivots[j]];
      }

      // Add the contributions to the stuf
      for (int j = 0; j < num_delayed; j++, jj++) {
        temp[jj] = x[delayed[j]];
      }
      for (int jp = contrib_ptr[ks]; jp < contrib_ptr[ks + 1]; jp++, jj++) {
        temp[jj] = x[contrib_rows[jp]];
      }
    }

    for (int ks = num_snodes - 1; ks >= 0; ks--) {
      // Get the pointers to the factor data
      int num_pivots, num_delayed;
      const int* pivots = nullptr;
      const int* delayed = nullptr;
      const T *L11, *L21;
      fact.get_factor(ks, &num_pivots, &num_delayed, &pivots, &delayed, &L11,
                      &L21);

      // Extract the variables from x
      int jj = 0;
      for (int j = 0; j < num_pivots; j++, jj++) {
        temp[jj] = x[pivots[j]];
      }

      // Add the contributions to the stuf
      for (int j = 0; j < num_delayed; j++, jj++) {
        temp[jj] = x[delayed[j]];
      }
      for (int jp = contrib_ptr[ks]; jp < contrib_ptr[ks + 1]; jp++, jj++) {
        temp[jj] = x[contrib_rows[jp]];
      }
    }
  }

 private:
  /**
   * @brief The contribution stack object used for the factorization
   */
  class ContributionStack {
   public:
    ContributionStack(int max_index, int max_work) {
      top_idx = 0;
      idx = new int[max_index];
      top_work = 0;
      work = new T[max_work];
    }
    ~ContributionStack() {
      delete[] idx;
      delete[] work;
    }

    /**
     * @brief Add the delayed pivots to the list of indices/vars
     *
     * @param nchildren Number of chiledren to look back at
     * @param fully_summed Initial number of fully summed variables
     * @param front_indices Indices in the front matrix
     * @param front_vars Front variables
     * @return Number of fully summed delayed pivots
     */
    int add_delayed_pivots(int nchildren, int fully_summed, int front_indices[],
                           int front_vars[]) {
      // Peak at the nchildren top entries
      int tmp_top = top_idx;

      for (int k = 0; k < nchildren; k++) {
        int delayed_pivots = idx[tmp_top - 2];
        int contrib_size = idx[tmp_top - 1];
        int* vars = &idx[tmp_top - 2 - contrib_size];

        for (int j = 0; j < delayed_pivots; j++) {
          int delayed = vars[j];
          if (front_indices[delayed] == -1) {
            front_indices[delayed] = fully_summed;
            front_vars[fully_summed] = delayed;
            fully_summed++;
          }
        }

        tmp_top -= (2 + contrib_size);
      }

      return fully_summed;
    }

    /**
     * @brief Push a contribution block onto the stack
     *
     * The matrix is arranged like this:
     *
     * F = [ F11  F12 ]
     *     [ F21  C   ]
     *
     * F11 is size num_pivots x num_pivots
     * C is size (front_size - num_pivots)
     *
     * @param num_pivots Number of columns selected as pivots
     * @param delayed_pivots Number of delayed pivots
     * @param front_size Size of the front matrix F
     * @param vars Variables on the front matrix
     * @param F The front matrix values
     */
    void push(int num_pivots, int delayed_pivots, int front_size,
              const int vars[], const T F[]) {
      // Copy the indices first
      int contrib_size = front_size - num_pivots;
      std::copy(vars + num_pivots, vars + front_size, &idx[top_idx]);
      top_idx += contrib_size;

      // Save the delayed pivots and size of the contribution block
      idx[top_idx] = delayed_pivots;
      idx[top_idx + 1] = contrib_size;
      top_idx += 2;

      // Copy the values into the data array
      for (int j = num_pivots; j < front_size; j++) {
        for (int i = num_pivots; i < front_size; i++, top_work++) {
          work[top_work] = F[i + front_size * j];
        }
      }
    }

    /**
     * @brief Pop a contribution block from the top of the stack
     *
     * @param delayed_pivots Number of delayed pivots
     * @param contrib_size Contribution block size
     * @param vars Indices for the contribution block
     * @param C The contribution block values
     */
    void pop(int* delayed_pivots, int* contrib_size, int* vars[], T* C[]) {
      *delayed_pivots = idx[top_idx - 2];
      int cb_size = idx[top_idx - 1];
      *contrib_size = cb_size;
      *vars = &idx[top_idx - 2 - cb_size];
      top_idx -= (2 + cb_size);

      *C = &work[top_work - cb_size * cb_size];
      top_work -= cb_size * cb_size;
    }

   private:
    int top_idx;   // Top of the index stack
    int* idx;      // Index/size values
    int top_work;  // Top of the entry stack
    T* work;       // Entries in the matrix
  };

  class MatrixFactor {
   public:
    MatrixFactor() {}
    ~MatrixFactor() {}

    void allocate(int ncols, int num_snodes, int factor_nnz) {}

    void reset() {}

    void add_factor(int ks, int num_pivots, const int pivots[], int num_delayed,
                    const int delayed[], int contrib_size, const T L[]) {}

    void get_factor(int ks, int* num_pivots, const int* pivots[],
                    int* num_delayed, const int* delayed[], const T* L11,
                    const T* L21);

   private:
    // int num_snodes;
    // int num_pivots;

    // int* pivots;
    // int* ptr;
  };

  /**
   * @brief Perform the multifrontal factorization
   *
   * Factor children that this front depends on
   * for children in front:
   *   factor_child(children);
   *
   * Find all the variables in this front
   *
   * Assemble the frontal matrix
   *
   * Pivot based on the fully summed nodes
   *
   * Compute the contribution block
   */
  int factor_numeric(const int ncols, const int colp[], const int rows[],
                     const T data[]) {
    int* temp = new int[2 * ncols];
    std::fill(temp, temp + 2 * ncols, -1);
    int* front_indices = temp;       // Indices in the front matrix
    int* front_vars = &temp[ncols];  // Variables in the front

    // TODO: Compute proper size for the frontal matrix
    int max_frontal_size = 10 * cholesky_nnz;
    T* F = new T[max_frontal_size];

    // TODO: Compute proper sizes for the stack
    int size = 10 * cholesky_nnz;
    ContributionStack stack(size, size);

    for (int ks = 0, k = 0; ks < num_snodes; k += snode_size[ks], ks++) {
      // Size of the super node
      int ns = snode_size[ks];

      // Number of children for this super node
      int nchildren = num_children[ks];

      // Get the frontal variables
      int fully_summed = 0, front_size = 0;
      get_frontal_vars(ks, k, ns, nchildren, stack, front_indices, front_vars,
                       &fully_summed, &front_size);

      // Assemble the frontal matrix
      assemble_front_matrix(k, ns, front_size, front_indices, colp, rows, data,
                            nchildren, stack, F);

      // Factor the frontal matrix and save the
      factor_front_matrix(ks, fully_summed, front_size, front_vars, F, stack,
                          fact);

      // Reset the front indices back to -1
      for (int j = 0; j < front_size; j++) {
        int var = front_vars[j];
        front_indices[var] = -1;
      }
    }

    // Clean up the data
    delete[] temp;

    return 0;
  }

  /**
   * @brief Get the variables for this front
   *
   * @param ks The super nodal index ks
   * @param k The offset into the super node variable list
   * @param ns The size of the super node
   * @param nchildren the number of children for this super node
   * @param stack The stack of contribution blocks
   * @param front_indices The front indices
   * @param front_vars The variables on the front
   * @param fully_summed Number of fully summed variables
   * @param front_size The front size
   */
  void get_frontal_vars(const int ks, const int k, const int ns,
                        const int nchildren, ContributionStack& stack,
                        int front_indices[], int front_vars[],
                        int* fully_summed, int* front_size) {
    // Set the ordering of the degrees of freeom in the front
    // Number of fully summed contributions (supernode pivots + delayed
    // pivots)
    for (int j = 0; j < ns; j++) {
      int var = snode_to_var[k + j];
      front_indices[var] = j;
      front_vars[j] = var;
    }

    // Add the additional contributions from the delayed pivots
    int f_summed =
        stack.add_delayed_pivots(nchildren, ns, front_indices, front_vars);

    // Get the entries predicted from Cholesky
    int start = contrib_ptr[ks];
    int cbsize = contrib_ptr[ks + 1] - start;
    for (int j = 0, *row = &contrib_rows[start]; j < cbsize; j++, row++) {
      front_indices[*row] = f_summed + j;
      front_vars[f_summed + j] = *row;
    }

    // Get the size of the front
    *fully_summed = f_summed;
    *front_size = f_summed + cbsize;
  }

  /**
   * @brief Assemble the frontal matrix associated with the delayed pivots and
   * super node entries
   *
   * @param k Offset into the super node list
   * @param ns Number of variables in this super node
   * @param front_size Front size
   * @param front_indices Front indices
   * @param colp Pointer into the column
   * @param rows Row indices
   * @param data Entries from the matrix
   * @param nchildren Number of children in the etree for this super node
   * @param stack Contribution stack
   * @param F The frontal matrix
   */
  void assemble_front_matrix(const int k, const int ns, int front_size,
                             const int front_indices[], const int colp[],
                             const int rows[], const T data[],
                             const int nchildren, ContributionStack& stack,
                             T F[]) {
    std::fill(F, F + front_size * front_size, 0.0);

    // Assemble the contributions into F from the matrix
    for (int j = 0; j < ns; j++) {
      // Get the column variable associated with the snode
      int var = snode_to_var[k + j];

      for (int ip = colp[var]; ip < colp[var + 1]; ip++) {
        // Get the
        int i = rows[ip];

        // Get the front index
        int ifront = front_indices[i];

        // Add the contribution to the frontal matrix
        if (ifront >= 0) {
          F[ifront + front_size * j] += data[ip];
        }
      }
    }

    // Add the contributions
    for (int child = 0; child < nchildren; child++) {
      int delayed_pivots;
      int contrib_size;
      int* contrib_indices;
      T* C;
      stack.pop(&delayed_pivots, &contrib_size, &contrib_indices, &C);

      // Add the contribution blocks
      for (int i = 0; i < contrib_size; i++) {
        int ifront = front_indices[contrib_indices[i]];

        for (int j = 0; j < contrib_size; j++) {
          int jfront = front_indices[contrib_indices[j]];

          F[ifront + front_size * jfront] += C[i + contrib_size * j];
        }
      }
    }
  }

  /**
   * @brief Factor the frontal matrix now that it is assembled.
   *
   * After factorization, push the contribution matrix to the stack and add the
   * factorization pieces
   *
   * @param ks The index of the super node
   * @param fully_summed The number of fully summed equations
   * @param front_size The front size
   * @param front_vars The front variables
   * @param F The frontal matrix itself
   * @param stack The stack for the contribution blocks
   * @param factor The factor contributions
   */
  void factor_front_matrix(const int ks, const int fully_summed,
                           const int front_size, const int front_vars[], T F[],
                           ContributionStack& stack, MatrixFactor& factor) {
    // Select the pivots
    int num_pivots = 0;

    // Perform the numerical factorization here..
    int num_delayed = fully_summed - num_pivots;
    stack.push(num_pivots, num_delayed, front_size, front_vars, F);

    // Push the factored matrix
    const int* pivots = front_vars;
    const int* delayed = &front_vars[num_pivots];
    factor.add_factor(ks, num_pivots, pivots, num_delayed, delayed,
                      front_size - fully_summed, F);
  }

  /**
   * @brief Perform the symbolic analysis phase on the non-zero matrix pattern
   *
   * This performs a post-order of the elimination tree, identifies super
   * nodes based on the post-ordering and performs a count of the numbers of
   * non-zero entries in the matrices.
   *
   * @param ncols Number of columns (equal to number of rows) in the matrix
   * @param colp Pointer into each column of the matrix
   * @param rows Row indices within each column of the matrix
   */
  void symbolic_analysis(const int ncols, const int colp[], const int rows[]) {
    // Allocate storage that we'll need
    int* work = new int[3 * ncols];

    // Compute the elimination tree
    int* parent = new int[ncols];
    compute_etree(ncols, colp, rows, parent, work);

    // Find the post-ordering for the elimination tree
    int* ipost = new int[ncols];
    post_order_etree(ncols, colp, rows, parent, ipost, work);

    // Count the column non-zeros in the post-ordering
    int* Lnz = new int[ncols];
    count_column_nonzeros(ncols, colp, rows, ipost, parent, Lnz, work);

    // Count up the total number of non-zeros in the Cholesky factorization
    cholesky_nnz = 0;
    for (int i = 0; i < ncols; i++) {
      cholesky_nnz += Lnz[i];
    }

    // Use the work array as a temporary here
    int* post = work;
    for (int i = 0; i < ncols; i++) {
      post[ipost[i]] = i;
    }

    // Initialize the super nodes
    var_to_snode = new int[ncols];
    snode_to_var = new int[ncols];
    num_snodes =
        init_super_nodes(ncols, post, parent, Lnz, var_to_snode, snode_to_var);

    // Count up the size of each snode
    snode_size = new int[num_snodes];
    std::fill(snode_size, snode_size + num_snodes, 0);
    for (int i = 0; i < ncols; i++) {
      snode_size[var_to_snode[i]]++;
    }

    // Count the children of supernodes within the post-ordered elimination
    // tree
    num_children = new int[num_snodes];
    count_super_node_children(ncols, parent, num_snodes, var_to_snode,
                              num_children, work);

    // Count up the sizes of the contribution blocks
    contrib_ptr = new int[num_snodes + 1];
    contrib_ptr[0] = 0;
    for (int is = 0, i = 0; is < num_snodes; i += snode_size[is], is++) {
      int var = snode_to_var[i + snode_size[is] - 1];
      contrib_ptr[is + 1] = Lnz[var];
    }

    // Count up the contribution block pointer
    for (int i = 0; i < num_snodes; i++) {
      contrib_ptr[i + 1] += contrib_ptr[i];
    }

    // Fill in the rows in the contribution blocks
    contrib_rows = new int[contrib_ptr[num_snodes]];
    build_nonzero_pattern(ncols, colp, rows, parent, num_snodes, snode_size,
                          var_to_snode, snode_to_var, contrib_ptr, contrib_rows,
                          work);

    // Allocate the arrays within the factorization
    fact.allocate(ncols, num_snodes, cholesky_nnz);

    delete[] work;
    delete[] parent;
    delete[] ipost;
    delete[] Lnz;
  }

  /**
   * @brief Compute the elimination tree
   *
   * @param ncols Number of columns
   * @param colp Pointer into each column
   * @param rows Row indices in each column
   * @param parent The etree parent child array
   * @param ancestor Largest ancestor of each node
   */
  void compute_etree(const int ncols, const int colp[], const int rows[],
                     int parent[], int ancestor[]) {
    // Initialize the parent and ancestor arrays
    std::fill(parent, parent + ncols, -1);
    std::fill(ancestor, ancestor + ncols, -1);

    for (int k = 0; k < ncols; k++) {
      // Loop over the column of k
      int start = colp[k];
      int end = colp[k + 1];
      for (int ip = start; ip < end; ip++) {
        int i = rows[ip];

        while (i < k) {
          int tmp = ancestor[i];

          // Update the largest ancestor of i
          ancestor[i] = k;

          // We've reached the root of the previous tree,
          // set the parent of i to k
          if (tmp == -1) {
            parent[i] = k;
            break;
          }

          i = tmp;
        }
      }
    }
  }

  /**
   * @brief Post-order the elimination tree
   *
   * ipost[i] = j
   *
   * means that node i of the original tree is the j-th node of the
   * post-ordered tree
   *
   * @param ncols Number of columns
   * @param colp Pointer into each column
   * @param rows Row indices in each column
   * @param parent The etree parent child array
   * @param ipost The computed post order
   * @param work Work array of size 3 * ncols
   */
  void post_order_etree(const int ncols, const int colp[], const int rows[],
                        const int parent[], int ipost[], int work[]) {
    int* head = work;
    int* next = &work[ncols];
    int* stack = &work[2 * ncols];

    std::fill(head, head + ncols, -1);
    std::fill(next, next + ncols, -1);

    // Initialize the heads of each linked list
    for (int j = ncols - 1; j >= 0; j--) {
      if (parent[j] != -1) {
        next[j] = head[parent[j]];
        head[parent[j]] = j;
      }
    }

    for (int j = 0, k = 0; j < ncols; j++) {
      if (parent[j] == -1) {
        // Perform a depth first search starting from j which is a root
        // in the etree
        k = depth_first_search(j, k, head, next, ipost, stack);
      }
    }
  }

  /**
   * @brief Perform a depth first search from node j
   *
   * @param j The root node to start from
   * @param k The post-order index
   * @param head The head of each linked list
   * @param next The next child in the linked lists
   * @param ipost The post order ipost[origin node i] = post node j
   * @param stack The stack for the depth first search
   * @return int The final post-order index
   */
  int depth_first_search(int j, int k, int head[], const int next[],
                         int ipost[], int stack[]) {
    int last = 0;     // Last position on the tack
    stack[last] = j;  // Put node j on the stack

    while (last >= 0) {
      // Look at the top of the stack and find the top node p and
      // its child i
      int p = stack[last];
      int i = head[p];

      if (i == -1) {
        // No unordered children of p left in the list
        ipost[p] = k;
        k++;
        last--;
      } else {
        // Remove i from the children of p and add i to the
        // stack to continue the depth first search
        head[p] = next[i];
        last++;
        stack[last] = i;
      }
    }

    return k;
  }

  /**
   * @brief Build the elimination tree and compute the number of non-zeros in
   * each column.
   *
   * @param ncols The number of columns in the matrix
   * @param colp The pointer into each column
   * @param rows The row indices for each matrix entry
   * @param parent The elimination tree/forest
   * @param Lnz The number of non-zeros in each column
   * @param work Work array of size ncols
   */
  void count_column_nonzeros(const int ncols, const int colp[],
                             const int rows[], const int ipost[],
                             const int parent[], int Lnz[], int work[]) {
    int* flag = work;
    std::fill(Lnz, Lnz + ncols, 0);
    std::fill(flag, flag + ncols, -1);

    // Loop over the original ordering of the matrix
    for (int k = 0; k < ncols; k++) {
      flag[k] = k;

      // Loop over the k-th column of the original matrix
      int ip_end = colp[k + 1];
      for (int ip = colp[k]; ip < ip_end; ip++) {
        int i = rows[ip];

        if (i < k) {
          // Scan up the etree
          while (flag[i] != k) {
            Lnz[i]++;
            flag[i] = k;

            // Set the next parent
            i = parent[i];
          }
        }
      }
    }
  }

  /**
   * @brief Initialize the supernodes in the matrix
   *
   * The supernodes share the same column non-zero pattern
   *
   * @param ncols The number of columns in the matrix
   * @param post The etree post ordering
   * @param parent The etree parents
   * @param Lnz The number of non-zeros per variable
   * @param vtosn Variable to super node
   * @param sntov Super node to variable
   * @return int The number of super nodes
   */
  int init_super_nodes(int ncols, const int post[], const int parent[],
                       const int Lnz[], int vtosn[], int sntov[]) {
    // First find the supernodes
    int snode = 0;

    // Loop over subsequent numbers in the post-ordering
    for (int i = 0; i < ncols;) {
      int var = post[i];  // Get the original variable number

      // Set the super node
      vtosn[var] = snode;
      sntov[i] = var;
      i++;

      int next_var = post[i];
      while (i < ncols && parent[var] == next_var &&
             (Lnz[next_var] == Lnz[var] - 1)) {
        vtosn[next_var] = snode;
        var = next_var;
        sntov[i] = var;
        i++;
        if (i < ncols) {
          next_var = post[i];
        }
      }

      snode++;
    }

    return snode;
  }

  /**
   * @brief Count up the number of children for each super node
   *
   * @param ncols Number of columns
   * @param parent Parent pointer for the elimination tree
   * @param ns Number of super nodes
   * @param vtosn Variable to super node array
   * @param nchild Number of children (output)
   * @param work Work array - at least number of super nodes
   */
  void count_super_node_children(const int ncols, const int parent[],
                                 const int ns, const int vtosn[], int nchild[],
                                 int work[]) {
    int* snode_parent = work;
    std::fill(nchild, nchild + ns, 0);
    std::fill(snode_parent, snode_parent + ns, -1);

    // Set up the snode parents first
    for (int j = 0; j < ncols; j++) {
      int pj = parent[j];

      if (pj != -1) {
        int js = vtosn[j];
        int pjs = vtosn[pj];

        if (pjs != js) {
          snode_parent[js] = pjs;
        }
      }
    }

    // Count up the children within the post-ordered elmination tree
    for (int i = 0; i < ns; i++) {
      if (snode_parent[i] != -1) {
        nchild[snode_parent[i]]++;
      }
    }
  }

  /**
   * @brief Find the non-zero rows in the post-ordered column using the
   * elimination tree - this does not included delayed pivots
   *
   * Find the non-zero rows below the diagonal in a column of L.
   *
   * This utilizes the elimination tree
   *
   * @param colp Column pointer
   * @param rows Rows for each column
   * @param parent Parent in the elimination tree
   * @param row_count The number of indices found so far
   * @param row_indices The array of row indices
   * @param tag Tag for the visited nodes
   * @param flag Array of flags (tag must not be contained in flag initially)
   */
  void build_nonzero_pattern(const int ncols, const int colp[],
                             const int rows[], const int parent[], int sn,
                             int snsize[], const int vtosn[], const int sntov[],
                             const int cptr[], int cvars[], int work[]) {
    int* Lnz = work;
    int* flag = &work[ncols];
    int* snvar = &work[2 * ncols];

    std::fill(Lnz, Lnz + ncols, 0);
    std::fill(flag, flag + ncols, -1);

    // Find the last variable in each super node
    for (int ks = 0, k = 0; ks < sn; k += snsize[ks], ks++) {
      snvar[ks] = sntov[k + snsize[ks] - 1];
    }

    // Loop over the original ordering of the matrix
    for (int k = 0; k < ncols; k++) {
      flag[k] = k;

      // Loop over the k-th column of the original matrix
      int iptr_end = colp[k + 1];
      for (int iptr = colp[k]; iptr < iptr_end; iptr++) {
        int i = rows[iptr];

        if (i < k) {
          // Scan up the etree
          while (flag[i] != k) {
            // Get the super node from the variable
            int is = vtosn[i];

            // Find the last variable in this super node
            int ivar = snvar[is];

            // If this is the last variable, add this to the row
            if (ivar == i) {
              cvars[cptr[is] + Lnz[is]] = k;
              Lnz[is]++;
            }

            // Flag the node
            flag[i] = k;

            // Set the next parent
            i = parent[i];
          }
        }
      }
    }
  }

  // The matrix
  std::shared_ptr<CSRMat<T>> mat;

  // Stability factor
  double ustab;

  // Estimated pivot growth factor
  double pivot_growth;

  // Number of non-zeros in the Choleksy factorization
  int cholesky_nnz;

  // Number of super nodes in the matrix
  int num_snodes;

  // Size of the super nodes
  int* snode_size;

  // Go from var to super node or super node to variable
  int* var_to_snode;
  int* snode_to_var;

  // Number of children for each super node
  int* num_children;

  // Compute the contribution blocks sizes (without delayed pivots)
  int* contrib_ptr;
  int* contrib_rows;

  // Storage for the matrix factorization
  MatrixFactor fact;
};

}  // namespace amigo

#endif  // AMIGO_SPARSE_LDL_H