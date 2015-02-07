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
}

void CGEMPLP::init()
{	
	SGVector<int32_t> fg_var_sizes = m_fg->get_cardinalities();	
	m_factors = m_fg->get_factors();

	int32_t num_factors = m_factors->get_num_elements();
	m_clique_separators.resize(num_factors);

	// get all the separators
	for (int32_t i = 0; i < num_factors; i++)
	{
		CFactor* factor_i = dynamic_cast<CFactor*>(m_factors->get_element(i));
		SGVector<int32_t> clique_i = factor_i->get_variables();
		SG_UNREF(factor_i);

		for (int32_t j = i+1; j < num_factors; j++)
		{
			CFactor* factor_j = dynamic_cast<CFactor*>(m_factors->get_element(j));
			SGVector<int32_t> clique_j = factor_j->get_variables();
			SG_UNREF(factor_j);

			const int32_t k = find_separator_index(clique_i, clique_j);
			if (k < 0) continue;

			m_clique_separators[i].push_back(k);
			m_clique_separators[j].push_back(k);
		}
	}
	
	m_clique_inds_separators.resize(num_factors);
	m_msgs_from_cluster.resize(num_factors);
	m_theta_cluster.resize(num_factors);

	for (int32_t c = 0; c < num_factors; c++)
	{
		CFactor* factor_c = dynamic_cast<CFactor*>(m_factors->get_element(c));
		SGVector<int32_t> vars_c = factor_c->get_variables();
		m_clique_inds_separators[c].resize(m_clique_separators[c].size());
		m_msgs_from_cluster[c].resize(m_clique_separators[c].size());

		for (uint32_t s = 0; s < m_clique_separators[c].size(); s++)
		{
			SGVector<int32_t> curr_separator = m_all_separators[m_clique_separators[c][s]];
			SGVector<int32_t> inds_s(curr_separator.size());
			SGVector<int32_t> dims_array(curr_separator.size());
			
			for (int32_t i = 0; i < inds_s.size(); i++)
			{
				inds_s[i] = vars_c.find(curr_separator[i])[0];
				dims_array[i] = fg_var_sizes[curr_separator[i]];
			}
			
			// initialize indices of separators inside the cluster
			m_clique_inds_separators[c][s] = inds_s;
			
			// initialize messages from cluster and set it 0
			SGNDArray<float64_t> message(dims_array);
			message.set_const(0);
			m_msgs_from_cluster[c][s] = message; 
		}

		// initialize potential on cluster
		m_theta_cluster[c] = convert_message(factor_c);

		SG_UNREF(factor_c);
	}

	// initialize messages in separators and set it 0
	m_msgs_into_separators.resize(m_all_separators.size());
	
	for (uint32_t i = 0; i < m_all_separators.size(); i++)
	{
		SGVector<int32_t> vars_separator = m_all_separators[i];
		SGVector<int32_t> dims_array(vars_separator.size());
		
		for (int32_t j = 0; j < dims_array.size(); j++)
			dims_array[j] = fg_var_sizes[vars_separator[j]];
	
		SGNDArray<float64_t> curr_array(dims_array);
		curr_array.set_const(0);
		m_msgs_into_separators[i] = curr_array.clone();
	}	
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

	return message;
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
	
	float64_t last_obj = CMath::INFTY;
	
	// block coordinate desent, outer loop
	for (int32_t it = 0; it < m_param.m_max_iter; ++it)
	{
		// update message, iterate over all cliques
		for (int32_t c = 0; c < m_factors->get_num_elements(); c++)
			update_messages(c);
		
		// calculate the objective value
		float64_t obj = 0;
		int32_t max_at;

		for (uint32_t s = 0; s < m_msgs_into_separators.size(); s++)
		{
			obj += m_msgs_into_separators[s].max_element(max_at);

			if (m_all_separators[s].size() == 1)
				assignment[m_all_separators[s][0]] = max_at;
		}
		
		// get the value of the decoded solution
		float64_t int_val = 0;
	
		// iterates over factors
		for (int32_t c = 0; c < m_factors->get_num_elements(); c++)
		{
			CFactor* factor = dynamic_cast<CFactor*>(m_factors->get_element(c));
			SGVector<int32_t> vars = factor->get_variables();
			SGVector<int32_t> var_assignment(vars.size());
			
			for (int32_t i = 0; i < vars.size(); i++)
				var_assignment[i] = assignment[vars[i]];

			// add value from current factor
			int_val += m_theta_cluster[c].get_value(var_assignment);

			SG_UNREF(factor);
		}
		
		float64_t obj_del = last_obj - obj;
		float64_t int_gap = obj - int_val;

		SG_SDEBUG("Iter= %d Objective=%f ObjBest=%f ObjDel=%f Gap=%f \n", (it + 1), obj, int_val, obj_del, int_gap);

		if (obj_del < m_param.m_obj_del_thr)
			break;
		
		if (int_gap < m_param.m_int_gap_thr)
			break;

		last_obj = obj;
	}
	
	float64_t energy = m_fg->evaluate_energy(assignment);
	SG_DEBUG("fg.evaluate_energy(assignment) = %f\n", energy);

	return energy;
}

void CGEMPLP::update_messages(int32_t id_clique)
{
	CFactor* factor = dynamic_cast<CFactor*>(m_factors->get_element(id_clique));
	SGVector<int32_t> vars = factor->get_variables();
	SGVector<int32_t> cards = factor->get_cardinalities();
	SGNDArray<float64_t> lam_sum(cards);

	if (m_theta_cluster[id_clique].len_array == 0)
		lam_sum.set_const(0);
	else
		lam_sum = m_theta_cluster[id_clique].clone();

	int32_t num_separators = m_clique_separators[id_clique].size();
	vector<SGNDArray<float64_t> > lam_minus; // substract message: \lambda_s^{-c}(x_s)
	// \sum_{\hat{s}} \lambda_{\hat{s}}^{-c}(x_{\hat{s}}) + \theta_c(x_c)
	for (int32_t s = 0; s < num_separators; s++)
	{
		int32_t id_separator = m_clique_separators[id_clique][s];
		SGNDArray<float64_t> tmp(m_msgs_into_separators[id_separator]);
		tmp -= m_msgs_from_cluster[id_clique][s];
		
		lam_minus.push_back(tmp);
		
		if (vars.size() == (int32_t)m_clique_inds_separators[id_clique][s].size())
			lam_sum += tmp;
		else
		{
			SGNDArray<float64_t> tmp_expand(lam_sum.get_dimensions());
			tmp.expand(tmp_expand, m_clique_inds_separators[id_clique][s]);
			lam_sum += tmp_expand;
		}

		// take out the old incoming message: \lambda_{c \to s}(x_s)
		m_msgs_into_separators[id_separator] -= m_msgs_from_cluster[id_clique][s];
	}
	
	for (int32_t s = 0; s < num_separators; s++)
	{
		// maximazation: \max_{x_c} \sum_{\hat{s}} \lambda_{\hat{s}}^{-c}(x_{\hat{s}}) + \theta_c(x_c)
		SGNDArray<float64_t> lam_max(lam_minus[s].get_dimensions());
		max_in_subdimension(lam_sum, m_clique_inds_separators[id_clique][s], lam_max);
		int32_t id_separator = m_clique_separators[id_clique][s];
		// weighted sum
		lam_max *= 1.0/num_separators;
		m_msgs_from_cluster[id_clique][s] = lam_max.clone();
		m_msgs_from_cluster[id_clique][s] -= lam_minus[s];
		// put in new message
		m_msgs_into_separators[id_separator] += m_msgs_from_cluster[id_clique][s];
	}

	SG_UNREF(factor);	
}

void CGEMPLP::max_in_subdimension(SGNDArray<float64_t> tar_arr, SGVector<int32_t> &subset_inds, SGNDArray<float64_t> &max_res) const
{	
	// If the subset equals the target array then maximizing would
	// give us the target array (assuming there is no reordering)
	if (subset_inds.size() == tar_arr.num_dims)
	{
		max_res = tar_arr.clone();
		return;
	}
	else
		max_res.set_const(-CMath::INFTY);

	// Go over all values of the target array. For each check if its
	// value on the subset is larger than the current max
	SGVector<int32_t> inds_for_tar(tar_arr.num_dims);
	inds_for_tar.zero();

	for (int32_t vi = 0; vi < tar_arr.len_array; vi++)
	{
		int32_t y = 0;
		
		if (subset_inds.size() == 1)
			y = inds_for_tar[subset_inds[0]];
		else if (subset_inds.size() == 2)
		{
			int32_t ind1 = subset_inds[0];
			int32_t ind2 = subset_inds[1];
			y = inds_for_tar[ind1] * max_res.dims[1] + inds_for_tar[ind2];
		}
		max_res[y] = max(max_res[y], tar_arr.array[vi]);
		tar_arr.next_index(inds_for_tar);
	}
}
