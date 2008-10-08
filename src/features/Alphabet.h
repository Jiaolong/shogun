/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006-2008 Soeren Sonnenburg
 * Copyright (C) 2006-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CALPHABET__H__
#define _CALPHABET__H__

#include "base/SGObject.h"
#include "lib/Mathematics.h"
#include "lib/common.h"


/// Alphabet of charfeatures/observations
enum EAlphabet
{
	/// DNA - letters A,C,G,T,*,N,n
	DNA=0,

	/// RAWDNA - letters 0,1,2,3
	RAWDNA=1,

	/// RNA - letters A,C,G,U,*,N,n
	RNA=2,

	/// PROTEIN - letters a-z
	PROTEIN=3,

	/// ALPHANUM - [0-9a-z]
	ALPHANUM=5,

	/// CUBE - [1-6]
	CUBE=6,

	/// RAW BYTE - [0-255]
	RAWBYTE=7,

	/// IUPAC_NUCLEIC_ACID
	IUPAC_NUCLEIC_ACID=8,

	/// IUPAC_AMINO_ACID
	IUPAC_AMINO_ACID=9,

	/// NONE - type has no alphabet
	NONE=10,

	/// unknown alphabet
	UNKNOWN=11,
};


/** The class Alphabet implements an alphabet and utility functions, to remap
 * characters to more (bit-)efficient representations, check if a string is
 * valid, compute histograms etc.
 *
 * Currently supported alphabets are DNA, RAWDNA, RNA, PROTEIN, ALPHANUM, CUBE, RAW,
 * IUPAC_NUCLEIC_ACID and IUPAC_AMINO_ACID.
 *
 */
class CAlphabet : public CSGObject
{
	public:
		/** constructor
		 *
		 * @param alpha alphabet to use
		 * @param len len
		 */
		CAlphabet(CHAR* alpha, INT len);

		/** constructor
		 *
		 * @param alpha alphabet (type) to use
		 */
		CAlphabet(EAlphabet alpha);

		/** constructor
		 *
		 * @param alpha alphabet to use
		 */
		CAlphabet(CAlphabet* alpha);
		~CAlphabet();

		/** set alphabet and initialize mapping table (for remap)
		 *
		 * @param alpha new alphabet
		 */
		bool set_alphabet(EAlphabet alpha);

		/** get alphabet
		 *
		 * @return alphabet
		 */
		inline EAlphabet get_alphabet()
		{
			return alphabet;
		}

		/** get number of symbols in alphabet
		 *
		 * @return number of symbols
		 */
		inline INT get_num_symbols()
		{
			return num_symbols;
		}

		/** get number of bits necessary to store
		 * all symbols in alphabet
		 *
		 * @return number of necessary storage bits
		 */
		inline INT get_num_bits()
		{
			return num_bits;
		}

		/** remap element e.g translate ACGT to 0123
		 *
		 * @param c element to remap
		 * @return remapped element
		 */
		inline BYTE remap_to_bin(BYTE c)
		{
			return maptable_to_bin[c];
		}

		/** remap element e.g translate 0123 to ACGT
		 *
		 * @param c element to remap
		 * @return remapped element
		 */
		inline BYTE remap_to_char(BYTE c)
		{
			return maptable_to_char[c];
		}

		/// clear histogram
		void clear_histogram();

		/** make histogram for whole string
		 *
		 * @param p string
		 * @param len length of string
		 */
		void add_string_to_histogram(CHAR* p, LONG len);

		/** make histogram for whole string
		 *
		 * @param p string
		 * @param len length of string
		 */
		void add_string_to_histogram(BYTE* p, LONG len);

		/** make histogram for whole string
		 *
		 * @param p string
		 * @param len length of string
		 */
		void add_string_to_histogram(SHORT* p, LONG len);

		/** make histogram for whole string
		 *
		 * @param p string
		 * @param len length of string
		 */
		void add_string_to_histogram(WORD* p, LONG len);

		/** make histogram for whole string
		 *
		 * @param p string
		 * @param len length of string
		 */
		void add_string_to_histogram(INT* p, LONG len);

		/** make histogram for whole string
		 *
		 * @param p string
		 * @param len length of string
		 */
		void add_string_to_histogram(UINT* p, LONG len);

		/** make histogram for whole string
		 *
		 * @param p string
		 * @param len length of string
		 */
		void add_string_to_histogram(LONG* p, LONG len);

		/** make histogram for whole string
		 *
		 * @param p string
		 * @param len length of string
		 */
		void add_string_to_histogram(ULONG* p, LONG len);

		/** add element to histogram
		 *
		 * @param p element
		 */
		inline void add_byte_to_histogram(BYTE p)
		{
			histogram[(INT) p]++;
		}

		/// print histogram
		void print_histogram();

		/** get histogram
		 *
		 * @param h where the histogram will be stored
		 * @param len length of histogram
		 */
		inline void get_hist(LONG** h, INT* len)
		{
			INT hist_size=(1 << (sizeof(BYTE)*8));
			ASSERT(h && len);
			*h=(LONG*) malloc(sizeof(LONG)*hist_size);
			ASSERT(*h);
			*len=hist_size;
			ASSERT(*len);
			memcpy(*h, &histogram[0], sizeof(LONG)*hist_size);
		}

		/// get pointer to histogram
		inline const LONG* get_histogram()
		{
			return &histogram[0];
		}

		/** check whether symbols in histogram are valid in alphabet
		 * e.g. for DNA if only letters ACGT appear
		 *
		 * @param print_error if errors shall be printed
		 * @return if symbols in histogram are valid in alphabet
		 */
		bool check_alphabet(bool print_error=true);

		/** check whether symbols in histogram ALL fit in alphabet
		 *
		 * @param print_error if errors shall be printed
		 * @return if symbols in histogram ALL fit in alphabet
		 */
		bool check_alphabet_size(bool print_error=true);

		/** return number of symbols in histogram
		 *
		 * @return number of symbols in histogram
		 */
		INT get_num_symbols_in_histogram();

		/** return maximum value in histogram
		 *
		 * @return maximum value in histogram
		 */
		INT get_max_value_in_histogram();

		/** return number of bits required to store all symbols in
		 * histogram
		 *
		 * @return number of bits required to store all symbols in
		 *         histogram
		 */
		INT get_num_bits_in_histogram();

		/** return alphabet name
		 *
		 * @param alphabet alphabet type to get name from
		 */
		static const CHAR* get_alphabet_name(EAlphabet alphabet);

	protected:
		/** init map table */
		void init_map_table();

		/** copy histogram
		 *
		 * @param src alphabet to copy histogram from
		 */
		void copy_histogram(CAlphabet* src);

	public:
		/** B_A */
		static const BYTE B_A;
		/** B_C */
		static const BYTE B_C;
		/** B_G */
		static const BYTE B_G;
		/** B_T */
		static const BYTE B_T;
		/** MAPTABLE UNDEF */
		static const BYTE MAPTABLE_UNDEF;
		/** alphabet names */
		static const CHAR* alphabet_names[11];

	protected:
		/** alphabet */
		EAlphabet alphabet;
		/** number of symbols */
		INT num_symbols;
		/** number of bits */
		INT num_bits;
		/** valid chars */
		BYTE valid_chars[1 << (sizeof(BYTE)*8)];
		/** maptable to bin */
		BYTE maptable_to_bin[1 << (sizeof(BYTE)*8)];
		/** maptable to char */
		BYTE maptable_to_char[1 << (sizeof(BYTE)*8)];
		/** histogram */
		LONG histogram[1 << (sizeof(BYTE)*8)];
};
#endif
