#include "FilterTIG.hpp"
#include <cstdint>
#include <iostream>
#include <assert.h>
#include "numpy/ndarrayobject.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

// The following conversion functions are taken from OpenCV's cv2.cpp file inside modules/python/src2 folder.
static PyObject* opencv_error = 0;

static int failmsg(const char *fmt, ...)
{
    char str[1000];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    PyErr_SetString(PyExc_TypeError, str);
    return 0;
}

class PyAllowThreads
{
public:
    PyAllowThreads() : _state(PyEval_SaveThread()) {}
    ~PyAllowThreads()
    {
        PyEval_RestoreThread(_state);
    }
private:
    PyThreadState* _state;
};

class ScopedGILRelease
{
public:
    inline ScopedGILRelease()
    {
        m_thread_state = PyEval_SaveThread();
    }

    inline ~ScopedGILRelease()
    {
        PyEval_RestoreThread(m_thread_state);
        m_thread_state = NULL;
    }

private:
    PyThreadState * m_thread_state;
};

class PyEnsureGIL
{
public:
    PyEnsureGIL() : _state(PyGILState_Ensure()) {}
    ~PyEnsureGIL()
    {
        PyGILState_Release(_state);
    }
private:
    PyGILState_STATE _state;
};

#define ERRWRAP2(expr) \
try \
{ \
    PyAllowThreads allowThreads; \
    expr; \
} \
catch (const cv::Exception &e) \
{ \
    PyErr_SetString(opencv_error, e.what()); \
    return 0; \
}

static PyObject* failmsgp(const char *fmt, ...)
{
  char str[1000];

  va_list ap;
  va_start(ap, fmt);
  vsnprintf(str, sizeof(str), fmt, ap);
  va_end(ap);

  PyErr_SetString(PyExc_TypeError, str);
  return 0;
}

static size_t REFCOUNT_OFFSET = (size_t)&(((PyObject*)0)->ob_refcnt) +
    (0x12345678 != *(const size_t*)"\x78\x56\x34\x12\0\0\0\0\0")*sizeof(int);

static inline PyObject* pyObjectFromRefcount(const int* refcount)
{
    return (PyObject*)((size_t)refcount - REFCOUNT_OFFSET);
}

static inline int* refcountFromPyObject(const PyObject* obj)
{
    return (int*)((size_t)obj + REFCOUNT_OFFSET);
}

class NumpyAllocator : public MatAllocator
{
public:
    NumpyAllocator() {}
    ~NumpyAllocator() {}

    void allocate(int dims, const int* sizes, int type, int*& refcount,
                  uchar*& datastart, uchar*& data, size_t* step)
    {
        PyEnsureGIL gil;

        int depth = CV_MAT_DEPTH(type);
        int cn = CV_MAT_CN(type);
        const int f = (int)(sizeof(size_t)/8);
        int typenum = depth == CV_8U ? NPY_UBYTE : depth == CV_8S ? NPY_BYTE :
                      depth == CV_16U ? NPY_USHORT : depth == CV_16S ? NPY_SHORT :
                      depth == CV_32S ? NPY_INT : depth == CV_32F ? NPY_FLOAT :
                      depth == CV_64F ? NPY_DOUBLE : f*NPY_ULONGLONG + (f^1)*NPY_UINT;
        int i;
        npy_intp _sizes[CV_MAX_DIM+1];
        for( i = 0; i < dims; i++ )
        {
            _sizes[i] = sizes[i];
        }

        if( cn > 1 )
        {
            /*if( _sizes[dims-1] == 1 )
                _sizes[dims-1] = cn;
            else*/
                _sizes[dims++] = cn;
        }

        PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);

        if(!o)
        {
            CV_Error_(CV_StsError, ("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
        }
        refcount = refcountFromPyObject(o);

        npy_intp* _strides = PyArray_STRIDES(o);
        for( i = 0; i < dims - (cn > 1); i++ )
            step[i] = (size_t)_strides[i];
        datastart = data = (uchar*)PyArray_DATA(o);
    }

    void deallocate(int* refcount, uchar*, uchar*)
    {
        PyEnsureGIL gil;
        if( !refcount )
            return;
        PyObject* o = pyObjectFromRefcount(refcount);
        Py_INCREF(o);
        Py_DECREF(o);
    }
};

NumpyAllocator g_roicropping_numpyAllocator;

enum { ARG_NONE = 0, ARG_MAT = 1, ARG_SCALAR = 2 };

static int pyopencv_to(const PyObject* o, Mat& m, const char* name = "<unknown>", bool allowND=true)
{
    //NumpyAllocator g_numpyAllocator;
    if(!o || o == Py_None)
    {
        if( !m.data )
            m.allocator = &g_roicropping_numpyAllocator;
        return true;
    }

    if( !PyArray_Check(o) )
    {
        failmsg("%s is not a numpy array", name);
        return false;
    }

    int typenum = PyArray_TYPE(o);
    int type = typenum == NPY_UBYTE ? CV_8U : typenum == NPY_BYTE ? CV_8S :
               typenum == NPY_USHORT ? CV_16U : typenum == NPY_SHORT ? CV_16S :
               typenum == NPY_INT || typenum == NPY_LONG ? CV_32S :
               typenum == NPY_FLOAT ? CV_32F :
               typenum == NPY_DOUBLE ? CV_64F : -1;

    if( type < 0 )
    {
        failmsg("%s data type = %d is not supported", name, typenum);
        return false;
    }

    int ndims = PyArray_NDIM(o);
    if(ndims >= CV_MAX_DIM)
    {
        failmsg("%s dimensionality (=%d) is too high", name, ndims);
        return false;
    }

    int size[CV_MAX_DIM+1];
    size_t step[CV_MAX_DIM+1], elemsize = CV_ELEM_SIZE1(type);
    const npy_intp* _sizes = PyArray_DIMS(o);
    const npy_intp* _strides = PyArray_STRIDES(o);
    bool transposed = false;

    for(int i = 0; i < ndims; i++)
    {
        size[i] = (int)_sizes[i];
        step[i] = (size_t)_strides[i];
    }

    if( ndims == 0 || step[ndims-1] > elemsize ) {
        size[ndims] = 1;
        step[ndims] = elemsize;
        ndims++;
    }

    if( ndims >= 2 && step[0] < step[1] )
    {
        std::swap(size[0], size[1]);
        std::swap(step[0], step[1]);
        transposed = true;
    }

    if( ndims == 3 && size[2] <= CV_CN_MAX && step[1] == elemsize*size[2] )
    {
        ndims--;
        type |= CV_MAKETYPE(0, size[2]);
    }

    if( ndims > 2 && !allowND )
    {
        failmsg("%s has more than 2 dimensions", name);
        return false;
    }

    m = Mat(ndims, size, type, PyArray_DATA(o), step);

    if( m.data )
    {
        m.refcount = refcountFromPyObject(o);
        m.addref(); // protect the original numpy array from deallocation
                    // (since Mat destructor will decrement the reference counter)
    };
    m.allocator = &g_roicropping_numpyAllocator;

    if( transposed )
    {
        Mat tmp;
        tmp.allocator = &g_roicropping_numpyAllocator;
        transpose(m, tmp);
        m = tmp;
    }
    return true;
}

static PyObject* pyopencv_from(const Mat& m)
{
    if( !m.data )
        Py_RETURN_NONE;
    Mat temp, *p = (Mat*)&m;
    if(!p->refcount || p->allocator != &g_roicropping_numpyAllocator)
    {

        temp.allocator = &g_roicropping_numpyAllocator;

        m.copyTo(temp);
        p = &temp;
    }

    p->addref();

    return pyObjectFromRefcount(p->refcount);
}

FilterTIG::FilterTIG(){
	import_array();
}

void FilterTIG::update(PyObject* weights_arr){

	//Release GIL
	ScopedGILRelease scoped;

	Mat w1f;
	pyopencv_to(weights_arr, w1f);

	CV_Assert(w1f.cols * w1f.rows == D && w1f.type() == CV_32F && w1f.isContinuous());
	float b[D], residuals[D];
	memcpy(residuals, w1f.data, sizeof(float)*D);
	for (int i = 0; i < NUM_COMP; i++){
		float avg = 0;
		for (int j = 0; j < D; j++){
			b[j] = residuals[j] >= 0.0f ? 1.0f : -1.0f;
			avg += residuals[j] * b[j];
		}
		avg /= D;
		_coeffs1[i] = avg, _coeffs2[i] = avg*2, _coeffs4[i] = avg*4, _coeffs8[i] = avg*8;
		for (int j = 0; j < D; j++)
			residuals[j] -= avg*b[j];
		UINT64 tig = 0;
		for (int j = 0; j < D; j++)
			tig = (tig << 1) | (b[j] > 0 ? 1 : 0);
		_bTIGs[i] = tig;
	}
}

void FilterTIG::reconstruct(PyObject* weights_arr){

	//Release GIL
	ScopedGILRelease scoped;

	Mat w1f;
	pyopencv_to(weights_arr, w1f);

	w1f = Mat::zeros(8, 8, CV_32F);
	float *weight = (float*)w1f.data;
	for (int i = 0; i < NUM_COMP; i++){
		UINT64 tig = _bTIGs[i];
		for (int j = 0; j < D; j++)
			weight[j] += _coeffs1[i] * (((tig >> (63-j)) & 1) ? 1 : -1);
	}
}

// For a W by H gradient magnitude map, find a W-7 by H-7 CV_32F matching score map
// Please refer to my paper for definition of the variables used in this function
PyObject* FilterTIG::matchTemplate(PyObject* grad){

	//Release GIL
	ScopedGILRelease scoped;

	Mat mag1u;

	pyopencv_to(grad, mag1u);

	const int H = mag1u.rows, W = mag1u.cols;
	const Size sz(W+1, H+1); // Expand original size to avoid dealing with boundary conditions
	Mat_<INT64> Tig1 = Mat_<INT64>::zeros(sz), Tig2 = Mat_<INT64>::zeros(sz);
	Mat_<INT64> Tig4 = Mat_<INT64>::zeros(sz), Tig8 = Mat_<INT64>::zeros(sz);
	Mat_<byte> Row1 = Mat_<byte>::zeros(sz), Row2 = Mat_<byte>::zeros(sz);
	Mat_<byte> Row4 = Mat_<byte>::zeros(sz), Row8 = Mat_<byte>::zeros(sz);
	Mat_<float> scores(sz);
	for(int y = 1; y <= H; y++){ 
		const byte* G = mag1u.ptr<byte>(y-1);
		INT64* T1 = Tig1.ptr<INT64>(y); // Binary TIG of current row
		INT64* T2 = Tig2.ptr<INT64>(y);
		INT64* T4 = Tig4.ptr<INT64>(y);
		INT64* T8 = Tig8.ptr<INT64>(y);
		INT64* Tu1 = Tig1.ptr<INT64>(y-1); // Binary TIG of upper row
		INT64* Tu2 = Tig2.ptr<INT64>(y-1);
		INT64* Tu4 = Tig4.ptr<INT64>(y-1);
		INT64* Tu8 = Tig8.ptr<INT64>(y-1);
		byte* R1 = Row1.ptr<byte>(y);
		byte* R2 = Row2.ptr<byte>(y);
		byte* R4 = Row4.ptr<byte>(y);
		byte* R8 = Row8.ptr<byte>(y);
		float *s = scores.ptr<float>(y);
		for (int x = 1; x <= W; x++) {
			byte g = G[x-1];
			R1[x] = (R1[x-1] << 1) | ((g >> 4) & 1);
			R2[x] = (R2[x-1] << 1) | ((g >> 5) & 1);
			R4[x] = (R4[x-1] << 1) | ((g >> 6) & 1);
			R8[x] = (R8[x-1] << 1) | ((g >> 7) & 1);
			T1[x] = (Tu1[x] << 8) | R1[x];
			T2[x] = (Tu2[x] << 8) | R2[x];
			T4[x] = (Tu4[x] << 8) | R4[x];
			T8[x] = (Tu8[x] << 8) | R8[x];
			s[x] = dot(T1[x], T2[x], T4[x], T8[x]);
		}
	}
	Mat matchCost1f;
	scores(Rect(8, 8, W-7, H-7)).copyTo(matchCost1f);
	return pyopencv_from(matchCost1f);
}

struct pixel_item {
  float score;
  Point point;
} ;

bool pixel_item_cmp (pixel_item* i,pixel_item* j) { return (i->score>j->score); }

PyObject* FilterTIG::nonMaxSup(PyObject* match_map, int NSS, int maxPoint, bool fast)
{

	pixel_item* item;
	vector<pixel_item*> valPnt;
	vector<pixel_item*> matchCost;

	Mat matchCost1f;
	pyopencv_to(match_map, matchCost1f);

	const int _h = matchCost1f.rows, _w = matchCost1f.cols;
	Mat isMax1u = Mat::ones(_h, _w, CV_8U), costSmooth1f;
	//ValStructVec<float, Point> valPnt;
	//matchCost.reserve(_h * _w);
	//valPnt.reserve(_h * _w);
	if (fast){
		blur(matchCost1f, costSmooth1f, Size(3, 3));
		for (int r = 0; r < _h; r++){
			const float* d = matchCost1f.ptr<float>(r);
			const float* ds = costSmooth1f.ptr<float>(r);
			for (int c = 0; c < _w; c++)
				if (d[c] >= ds[c]){
					//valPnt.pushBack(d[c], Point(c, r));
					item = new pixel_item;
					if(!item)
						return Py_None;
					item->score = d[c];
					item->point.x = c;
					item->point.y = r;
					valPnt.push_back(item);
				}
		}
	}
	else{
		for (int r = 0; r < _h; r++){
			const float* d = matchCost1f.ptr<float>(r);
			for (int c = 0; c < _w; c++){
				//valPnt.pushBack(d[c], Point(c, r));
				item = new pixel_item;
				if(!item)
					return Py_None;
				item->score = d[c];
				item->point.x = c;
				item->point.y = r;
				valPnt.push_back(item);
			}
		}
	}

	sort( valPnt.begin(), valPnt.end(), pixel_item_cmp);
	for (int i = 0; i < valPnt.size(); i++){
		Point pnt = valPnt[i]->point;
		if (isMax1u.at<byte>(pnt)){
			item = new pixel_item;
			if(!item)
				return Py_None;
			item->score = valPnt[i]->score;
			item->point.x = pnt.x;
			item->point.y = pnt.y;
			matchCost.push_back(item);
			for (int dy = -NSS; dy <= NSS; dy++) for (int dx = -NSS; dx <= NSS; dx++){
				Point neighbor = pnt + Point(dx, dy);
				if (!CHK_IND(neighbor))
					continue;
				isMax1u.at<byte>(neighbor) = false;
			}
		}
		if (matchCost.size() >= maxPoint)
			break;
	}

	PyObject *pyMatchCost = PyList_New(0);
	if(!pyMatchCost)
		return Py_None;

	PyObject* py_point;
	PyObject* py_item;
	for (int i = 0; i < matchCost.size(); i++){
		py_point = Py_BuildValue("(ii)", matchCost[i]->point.x, matchCost[i]->point.y);
		py_item = Py_BuildValue("(Of)", py_point, matchCost[i]->score);
		PyList_Append(pyMatchCost, py_item);
	}

	return pyMatchCost;
}
