#pragma once

#include <Python.h>

#ifdef __WIN32
    #define INT64 long long
#else
    #define INT64 long
	typedef unsigned long UINT64;
#endif

#define CHK_IND(p) ((p).x >= 0 && (p).x < _w && (p).y >= 0 && (p).y < _h)

typedef unsigned char byte;

class FilterTIG
{
public:
	void update(PyObject* weights_arr);

	// For a W by H gradient magnitude map, find a W-7 by H-7 CV_32F matching score map
	PyObject* matchTemplate(PyObject* grad);
	PyObject* nonMaxSup(PyObject* match_map, int NSS = 1, int maxPoint = 50, bool fast = true);
	
	inline float dot(const INT64 tig1, const INT64 tig2, const INT64 tig4, const INT64 tig8);

public:
	FilterTIG();
	void reconstruct(PyObject* weights_arr); // For illustration purpose

private:
	static const int NUM_COMP = 2; // Number of components
	static const int D = 64; // Dimension of TIG
	INT64 _bTIGs[NUM_COMP]; // Binary TIG features
	float _coeffs1[NUM_COMP]; // Coefficients of binary TIG features

	// For efficiently deals with different bits in CV_8U gradient map
	float _coeffs2[NUM_COMP], _coeffs4[NUM_COMP], _coeffs8[NUM_COMP]; 
};


inline float FilterTIG::dot(const INT64 tig1, const INT64 tig2, const INT64 tig4, const INT64 tig8)
{
    INT64 bcT1 = __builtin_popcountll(tig1);
    INT64 bcT2 = __builtin_popcountll(tig2);
    INT64 bcT4 = __builtin_popcountll(tig4);
    INT64 bcT8 = __builtin_popcountll(tig8);
	
    INT64 bc01 = (__builtin_popcountll(_bTIGs[0] & tig1) << 1) - bcT1;
    INT64 bc02 = ((__builtin_popcountll(_bTIGs[0] & tig2) << 1) - bcT2) << 1;
    INT64 bc04 = ((__builtin_popcountll(_bTIGs[0] & tig4) << 1) - bcT4) << 2;
    INT64 bc08 = ((__builtin_popcountll(_bTIGs[0] & tig8) << 1) - bcT8) << 3;

    INT64 bc11 = (__builtin_popcountll(_bTIGs[1] & tig1) << 1) - bcT1;
    INT64 bc12 = ((__builtin_popcountll(_bTIGs[1] & tig2) << 1) - bcT2) << 1;
    INT64 bc14 = ((__builtin_popcountll(_bTIGs[1] & tig4) << 1) - bcT4) << 2;
    INT64 bc18 = ((__builtin_popcountll(_bTIGs[1] & tig8) << 1) - bcT8) << 3;

	return _coeffs1[0] * (bc01 + bc02 + bc04 + bc08) + _coeffs1[1] * (bc11 + bc12 + bc14 + bc18);
}
