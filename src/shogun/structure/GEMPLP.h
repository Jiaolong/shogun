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
#include "./mplp/mplp_alg.h"

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
	 * @param param parameters
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
	 * @param assignment the assignment
	 * @return the total energy after doing inference
	 */
	virtual float64_t inference(SGVector<int32_t> assignment);

private:
	/** Initialize GEMPLP with factor graph */
	void init();

public:
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
	
	void exist_or_insert(vector<int32_t>& v, int32_t k);

public:
	/** GEMPLP parameter */
	Parameter m_param;
	/** all factors in the graph*/
	CDynamicObjectArray* m_factors;
	/** all separators */
	vector<SGVector<int32_t> > m_all_separators;

	vector<vector<int> > m_all_intersects;
	vector<Region> m_all_regions;
	vector<vector<int> > m_all_region_inds;	
	vector<MulDimArr> m_sum_into_intersects;
	vector<vector<int> > m_all_region_intersects;
	vector<int> m_var_sizes;
	vector<int> m_decoded_res;

	MPLPAlg* m_mplp;
};
}
#endif
