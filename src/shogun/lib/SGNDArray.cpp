/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Written (W) 2010,2012 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 * Copyright (C) 2012 Soeren Sonnenburg
 */

#include <shogun/lib/common.h>
#include <shogun/lib/SGNDArray.h>
#include <shogun/lib/SGReferencedData.h>

namespace shogun
{

template<class T> SGNDArray<T>::SGNDArray() :
	SGReferencedData()
{
	init_data();
}

template<class T> SGNDArray<T>::SGNDArray(T* a, index_t* d, index_t nd, bool ref_counting) :
	SGReferencedData(ref_counting)
{
	array = a;
	dims = d;
	num_dims = nd;
}

template<class T> SGNDArray<T>::SGNDArray(index_t* d, index_t nd, bool ref_counting) :
	SGReferencedData(ref_counting), dims(d), num_dims(nd)
{
	int64_t total = 1;
	for (int32_t i=0; i<num_dims; i++)
		total *= dims[i];
	ASSERT(total>0)
	array = SG_MALLOC(T, total);
}

template<class T> SGNDArray<T>::SGNDArray(const SGNDArray &orig) :
	SGReferencedData(orig)
{
	copy_data(orig);
}

template<class T> SGNDArray<T>::~SGNDArray()
{
	unref();
}

template<class T> void SGNDArray<T>::copy_data(const SGReferencedData &orig)
{
	array = ((SGNDArray*)(&orig))->array;
	dims = ((SGNDArray*)(&orig))->dims;
	num_dims = ((SGNDArray*)(&orig))->num_dims;
}

template<class T> void SGNDArray<T>::init_data()
{
	array = NULL;
	dims = NULL;
	num_dims = 0;
}

template<class T> void SGNDArray<T>::free_data()
{
	SG_FREE(array);
	SG_FREE(dims);

	array     = NULL;
	dims      = NULL;
	num_dims  = 0;
}

template<class T> void SGNDArray<T>::transpose_matrix(index_t matIdx) const
{
	ASSERT(array && dims && num_dims > 2 && dims[2] > matIdx)

	T aux;
	// Index to acces directly the elements of the matrix of interest
	int64_t idx = int64_t(matIdx)*int64_t(dims[0])*dims[1];

	for (int64_t i=0; i<dims[0]; i++)
		for (int64_t j=0; j<i-1; j++)
		{
			aux = array[idx + i + j*dims[0]];
			array[idx + i + j*dims[0]] = array[idx + j + i*dims[0]];
			array[idx + j + i*dims[1]] = aux;
		}

	// Swap the sizes of the two first dimensions
	index_t auxDim = dims[0];
	dims[0] = dims[1];
	dims[1] = auxDim;
}

template<class T> SGNDArray<T>& SGNDArray<T>::operator=(const SGNDArray& ndarray)
{
	copy_data(ndarray);
	
	return (*this);
}

template<class T> SGNDArray<T>& SGNDArray<T>::operator=(T val)
{
	for (index_t i = 0; i < get_size(); i++)
	{
		array[i] = val;
	}
	
	return (*this);
}

template<>
SGNDArray<float64_t>& SGNDArray<float64_t>::operator*=(float64_t val)
{
	for (index_t i = 0; i < get_size(); i++)
	{
		array[i] *= val;
	}
	
	return (*this);
}

template<>
SGNDArray<float64_t>& SGNDArray<float64_t>::operator+=(SGNDArray& ndarray)
{
	ASSERT(get_size() == ndarray.get_size());
	ASSERT(num_dims == ndarray.num_dims);
	
	for (index_t i = 0; i < get_size(); i++)
	{
		array[i] += ndarray.array[i];
	}
	
	return (*this);
}

template<>
SGNDArray<float64_t>& SGNDArray<float64_t>::operator-=(SGNDArray& ndarray)
{
	ASSERT(get_size() == ndarray.get_size());
	ASSERT(num_dims == ndarray.num_dims);
	
	for (index_t i = 0; i < get_size(); i++)
	{
		array[i] -= ndarray.array[i];
	}
	
	return (*this);
}
	
template<>
float64_t SGNDArray<float64_t>::max_element(int32_t &max_at)
{
	float64_t m = array[0];
	max_at = 0;
	
	for (int32_t i = 1; i < get_size(); i++)
	{
		if (array[i] >= m)
		{
			max_at = i;
			m = array[i];
		}
	}
	
	return m;
}

template<class T>
T SGNDArray<T>::get_value(index_t *index) const
{
	int32_t y = 0;
	int32_t fact = 1;
	
	for (int32_t i = num_dims - 1; i >= 0; i--)
	{
		y += index[i] * fact;
		fact *= dims[i];
	}
	
	return array[y];
}

template<class T>
void SGNDArray<T>::next_index(index_t *curr_index) const
{
	for (int32_t i = num_dims - 1; i >= 0; i++ )
	{
		curr_index[i]++;
		if (curr_index[i] < dims[i])
		{
			break;
		}
		curr_index[i] = 0;
	}
}

template<>
void SGNDArray<float64_t>::expand(SGNDArray &big_array, index_t *axes, int32_t num_axes)
{
	// TODO: A nice implementation would be a function like repmat in matlab
	REQUIRE(num_axes <= 2, "Only 1-d and 2-d array can be expanded currently.");
	// Initialize indices in big array to zeros
	index_t inds_big[big_array.num_dims];
	for (int32_t i = 0; i < big_array.num_dims; i++)
	{
		inds_big[i] = 0;
	}

	// Replicate the small array to the big one.
	// Go over the big one by one and take the corresponding value
	float64_t* data_big = &big_array.array[0];
	for (int32_t vi = 0; vi < big_array.get_size(); vi++)
	{
		int32_t y = 0;

		if (num_axes == 1)
		{
			y = inds_big[axes[0]];
		}
		else if (num_axes == 2)
		{
			int32_t ind1 = axes[0];
			int32_t ind2 = axes[1];
			y = inds_big[ind1] * dims[1] + inds_big[ind2];
		}

		*data_big = array[y];
		data_big++;

		// Move to the next index
		big_array.next_index(inds_big);
	}
}

template class SGNDArray<bool>;
template class SGNDArray<char>;
template class SGNDArray<int8_t>;
template class SGNDArray<uint8_t>;
template class SGNDArray<int16_t>;
template class SGNDArray<uint16_t>;
template class SGNDArray<int32_t>;
template class SGNDArray<uint32_t>;
template class SGNDArray<int64_t>;
template class SGNDArray<uint64_t>;
template class SGNDArray<float32_t>;
template class SGNDArray<float64_t>;
template class SGNDArray<floatmax_t>;
}
