/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Jiaolong Xu
 * Copyright (C) 2014 Jiaolong Xu
 */

#ifndef __GEMPLP_H__
#define __GEMPLP_H__

#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/structure/FactorGraph.h>
#include <shogun/structure/Factor.h>
#include <shogun/structure/MAPInference.h>
#include <shogun/lib/SGNDArray.h>

#include <vector>

using namespace std;

namespace shogun
{
#define IGNORE_IN_CLASSLIST
/** GEMPLP (Generalized Max-product LP Relaxation) inference for fatcor graph
 *
 * Please refer to following paper for more details:
 *
 * [1] Fixing max-product: Convergent message passing algorithms for MAP LP-relaxations
 * Amir Globerson, Tommi Jaakkola
 * Advances in Neural Information Processing Systems (NIPS). Vancouver, Canada. 2007.
 *
 * [2] Approximate Inference in Graphical Models using LP Relaxations.
 * David Sontag
 * Ph.D. thesis, Massachusetts Institute of Technology, 2010.
 *
 * The original implementation of GEMPLP can be found:
 * http://cs.nyu.edu/~dsontag/code/mplp_ver2.tgz
 * http://cs.nyu.edu/~dsontag/code/mplp_ver1.tgz
 */
IGNORE_IN_CLASSLIST class CGEMPLP: public CMAPInferImpl
{
public:
	/** Parameter for GEMPLP */
	struct Parameter
	{
		Parameter(const int32_t max_iter = 1000,
		          const float64_t obj_del_thr = 0.0002,
		          const float64_t int_gap_thr = 0.0002)
			: m_max_iter(max_iter),
			  m_obj_del_thr(obj_del_thr),
			  m_int_gap_thr(int_gap_thr)
		{}

		/** maximum number of outer iterations*/
		int32_t m_max_iter;
		/** threshold of the delta objective value */
		float64_t m_obj_del_thr;
		/** threshold of objective gap betweeb current and best integer assignment */
		float64_t m_int_gap_thr;
	};

public:
	/** Constructor */
	CGEMPLP();

	/** Constructor
	 *
	 * @param fg factor graph
	 */
	CGEMPLP(CFactorGraph* fg, Parameter param = Parameter());

	/** Destructor */
	virtual ~CGEMPLP();

	/** @return class name */
	virtual const char* get_name() const
	{
		return "GEMPLP";
	}

	/** Inference
	 *
	 * @param the assignment
	 * @return the total energy after doing inference
	 */
	virtual float64_t inference(SGVector<int32_t> assignment);

private:
	/** Initialize GEMPLP with factor graph */
	void init();
	
	/** Message updating on a clique
	 *
	 * Please refer to "GEMPLP" in NIPS paper of
	 * A. Globerson and T. Jaakkola [1] for more details.
	 *	
	 */
	void update_messages(int32_t id_clique);	

public:
	/** Computer the maximum value along the sub-dimension
	 *
	 * @param tar_arr target nd array
	 * @param subset_inds sub-dimension indices
	 * @param max_res the result nd array
	 */
	void max_in_subdimension(SGNDArray<float64_t> tar_arr, SGVector<int32_t> &subset_inds, SGNDArray<float64_t> &max_res) const;

	/** Find separators between clqiues
	 *
	 * @param clique_A clique A
	 * @param clique_B clique B
	 * @return index in all separators
	 */
	int32_t find_separator_index(SGVector<int32_t> clique_A, SGVector<int32_t> clique_B);

	/** Convert original energies to potentials of the cluster
	 * GEMPLP objective function is a maximazation function, we use - energy
	 * as potential. The indices of the table factor energy also need to be
	 * converted to the order of the nd array.
	 *
	 * @param factor factor which contains energy
	 * @return potential of the cluster in MPLP
	 */
	SGNDArray<float64_t> convert_message(CFactor* factor);

public:
	/** GEMPLP parameter */
	Parameter m_param;
	/** all factors in the graph*/
	CDynamicObjectArray* m_factors;
	/** all separators */
	vector<SGVector<int32_t> > m_all_separators;
	/** the separator indices (node indices) on each clique */
	vector<vector<int32_t> > m_clique_separators;
	/** the indices (orders in the clique) of the separators on each clique */
	vector<vector<SGVector<int32_t> > > m_clique_inds_separators;
	/** store the sum of messages into separators */
	vector<SGNDArray<float64_t> > m_msgs_into_separators;
	/** store the messages from cluster to separator */
	vector<vector<SGNDArray<float64_t> > > m_msgs_from_cluster;
	/** store the original (-) energy of the factors */
	vector<SGNDArray<float64_t> > m_theta_cluster;
	/** current assignment */
	SGVector<int32_t> m_current_assignment;
};
}
#endif
