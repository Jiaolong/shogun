/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Jiaolong Xu
 * Copyright (C) 2014 Jiaolong Xu
 */

#include <shogun/structure/GEMPLP.h>
#include <shogun/io/SGIO.h>
#include <algorithm>

using namespace shogun;
using namespace std;

CGEMPLP::CGEMPLP()
	: CMAPInferImpl()
{
	m_fg = NULL;
	m_factors = NULL;
}

CGEMPLP::CGEMPLP(CFactorGraph* fg, Parameter param)
	: CMAPInferImpl(fg),
	  m_param(param)
{
	ASSERT(m_fg != NULL);

	init();
}

CGEMPLP::~CGEMPLP()
{
	if(m_factors != NULL)
		SG_UNREF(m_factors);

	if(m_mplp != NULL)
		delete m_mplp;
}

void CGEMPLP::init()
{	
	SGVector<int32_t> fg_var_sizes = m_fg->get_cardinalities();	
	m_factors = m_fg->get_factors();

	int32_t num_factors = m_factors->get_num_elements();
		
	m_var_sizes.resize(fg_var_sizes.size());
	for (int32_t i = 0; i < fg_var_sizes.size(); i++)
		m_var_sizes[i] = fg_var_sizes[i];

	m_all_region_intersects.resize(num_factors);
	
	// get all the separators
	for (int32_t i = 0; i < num_factors; i++)
	{
		CFactor* factor_i = dynamic_cast<CFactor*>(m_factors->get_element(i));
		SGVector<int32_t> clique_i = factor_i->get_variables();
		SG_UNREF(factor_i);

		for (int32_t j = i; j < num_factors; j++)
		{
			CFactor* factor_j = dynamic_cast<CFactor*>(m_factors->get_element(j));
			SGVector<int32_t> clique_j = factor_j->get_variables();
			SG_UNREF(factor_j);

			const int32_t k = find_separator_index(clique_i, clique_j);
			if (k < 0) continue;

			exist_or_insert(m_all_region_intersects[i], k);
			
			if (j != i)
				exist_or_insert(m_all_region_intersects[j], k);
		}
	}
	
	m_all_intersects.resize(m_all_separators.size());
	
	for (int32_t i = 0; i < m_all_separators.size(); i++)
		for (int32_t j = 0; j < m_all_separators[i].size(); j++)
			m_all_intersects[i].push_back(m_all_separators[i][j]);	

	m_all_region_inds.resize(num_factors);

	for (int32_t c = 0; c < num_factors; c++)
	{
		CFactor* factor_c = dynamic_cast<CFactor*>(m_factors->get_element(c));
		SGVector<int32_t> vars_c = factor_c->get_variables();
		m_all_region_inds[c].resize(vars_c.size());

		for (int32_t v = 0; v < vars_c.size(); v++)
			m_all_region_inds[c][v] = vars_c[v];

		// initialize region on cluster
		SGNDArray<float64_t> nd_array = convert_message(factor_c);
		MulDimArr *curr_lambda;

		if (nd_array.len_array != 0)
		{
			SGVector<int32_t> dims_nd = nd_array.get_dimensions();
			vector<int32_t> dims_array(dims_nd.size());
			for (int32_t d = 0; d < dims_nd.size(); d++)
				dims_array[d] = dims_nd[d];

			curr_lambda = new MulDimArr(dims_array);
		  	
			for (int32_t i = 0; i < curr_lambda->m_n_prodsize; i++)
		    	(*curr_lambda)[i] = nd_array.array[i];
		}
		else
		  curr_lambda = new MulDimArr; // Empty constructor

		// Initialize current region
		Region curr_region(m_all_region_inds[c], *curr_lambda, m_all_intersects, m_all_region_intersects[c], m_var_sizes);
		
		delete curr_lambda;
		
		m_all_regions.push_back(curr_region);

		SG_UNREF(factor_c);
	}

	// initialize messages in separators and set it 0
	m_sum_into_intersects.clear();
	
	for (uint32_t i = 0; i < m_all_intersects.size(); i++)
	{
		vector<int32_t> vars_separator = m_all_intersects[i];
		vector<int32_t> dims_array(vars_separator.size());
		
		for (int32_t j = 0; j < dims_array.size(); j++)
			dims_array[j] = fg_var_sizes[vars_separator[j]];
		
		MulDimArr curr_suminto(dims_array);
		curr_suminto = 0;
		m_sum_into_intersects.push_back(curr_suminto);
	}

	// Initialize output vector
	m_decoded_res.clear();
	for (int32_t i = 0; i < m_var_sizes.size(); i++)
		m_decoded_res.push_back(0);

	m_mplp = new MPLPAlg(); 
	m_mplp->m_all_intersects = m_all_intersects;
	m_mplp->m_all_regions = m_all_regions;
	m_mplp->m_all_region_inds =  m_all_region_inds;	
	m_mplp->m_sum_into_intersects =  m_sum_into_intersects;
	m_mplp->m_all_region_intersects = m_all_region_intersects;
	m_mplp->m_var_sizes =  m_var_sizes;
	m_mplp->m_decoded_res = m_decoded_res;
}

void CGEMPLP::exist_or_insert(vector<int32_t>& v, int32_t k)
{
	for (uint32_t i = 0; i < v.size(); i++)
		if (v[i] == k)	return;

	v.push_back(k);
}

SGNDArray<float64_t> CGEMPLP::convert_message(CFactor* factor)
{
	SGVector<float64_t> energies = factor->get_energies();
	SGVector<int32_t> cards = factor->get_cardinalities();

	SGNDArray<float64_t> message(cards);

	if (cards.size() == 1)
	{
		for (int32_t i = 0; i < energies.size(); i++)
			message.array[i] = - energies[i];
	}
	else if (cards.size() == 2)
	{
		for (int32_t y = 0; y < cards[1]; y++)
			for (int32_t x = 0; x < cards[0]; x++)
				message.array[x*cards[1]+y] = - energies[y*cards[0]+x];
	}
	else
		SG_ERROR("Index issue has not been solved for higher order (>=3) factors.");

	return message.clone();
}

int32_t CGEMPLP::find_separator_index(SGVector<int32_t> clique_A, SGVector<int32_t> clique_B)
{
	vector<int32_t> tmp;

	for (int32_t i = 0; i < clique_A.size(); i++)
	{
		for (int32_t j = 0; j < clique_B.size(); j++)
		{
			if (clique_A[i] == clique_B[j])
				tmp.push_back(clique_A[i]);
		}
	}
	
	// return -1 if intersetion is empty
	if (tmp.size() == 0)	return -1;

	
	SGVector<int32_t> sAB(tmp.size());
	for (uint32_t i = 0; i < tmp.size(); i++)
		sAB[i] = tmp[i];

	// find (or add) separator set
	int32_t k;
	for (k = 0; k < (int32_t)m_all_separators.size(); k++)
		if (m_all_separators[k].equals(sAB))
			break;

	if (k == (int32_t)m_all_separators.size())
		m_all_separators.push_back(sAB);
	
	return k;
}

float64_t CGEMPLP::inference(SGVector<int32_t> assignment)
{
	REQUIRE(assignment.size() == m_fg->get_cardinalities().size(),
	        "%s::inference(): the output assignment should be prepared as"
	        "the same size as variables!\n", get_name());

	// iterate over message loop
	SG_SDEBUG("Running MPLP for %d iterations\n",  m_param.m_max_iter);
	
	m_mplp->RunMPLP(m_param.m_max_iter, m_param.m_obj_del_thr, m_param.m_int_gap_thr);
	
	for (int32_t i = 0; i < assignment.size(); i++)
		assignment[i] = m_mplp->m_decoded_res[i];
	
	float64_t energy = m_fg->evaluate_energy(assignment);
	SG_DEBUG("fg.evaluate_energy(assignment) = %f\n", energy);

	return energy;
}
