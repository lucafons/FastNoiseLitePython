//
// Created by Luca Fonstad on 1/7/23.
//
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <iostream>
#include <Python.h>
#include <memory>
#include <vector>
#include "numpy/arrayobject.h"
#include "structmember.h"
#include "FastNoiseLite.h"

using namespace std;

typedef struct {
    PyObject_HEAD
    unique_ptr<FastNoiseLite> noise;
    PyObject* noiseType;
    PyObject* fractalType;
    PyObject* octaves;
    PyObject* lacunarity;
    PyObject* gain;
    PyObject* weightedStrength;
    PyObject* pingPongStrength;
    PyObject* cellularDistanceFunction;
    PyObject* cellularReturnType;
    PyObject* cellularJitter;
    PyObject* domainWarpType;
    PyObject* domainWarpAmp;
    PyObject* rotationType3d;
} NoiseObject;

static void Noise_dealloc(NoiseObject *self) {
    Py_XDECREF(self->noiseType);
    Py_XDECREF(self->fractalType);
    Py_XDECREF(self->octaves);
    Py_XDECREF(self->lacunarity);
    Py_XDECREF(self->gain);
    Py_XDECREF(self->weightedStrength);
    Py_XDECREF(self->pingPongStrength);
    Py_XDECREF(self->cellularDistanceFunction);
    Py_XDECREF(self->cellularReturnType);
    Py_XDECREF(self->cellularJitter);
    Py_XDECREF(self->domainWarpAmp);
    Py_XDECREF(self->domainWarpType);
    Py_XDECREF(self->rotationType3d);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int Noise_init(NoiseObject* self, [[maybe_unused]] PyObject* args, [[maybe_unused]] PyObject* kwds) {
    self->noise = make_unique<FastNoiseLite>();
    self->noiseType = PyUnicode_FromString("OpenSimplex2");
    Py_INCREF(self->noiseType);
    self->cellularDistanceFunction = PyUnicode_FromString("EuclideanSq");
    Py_INCREF(self->cellularDistanceFunction);
    self->cellularReturnType = PyUnicode_FromString("Distance");
    Py_INCREF(self->cellularReturnType);
    self->domainWarpType = PyUnicode_FromString("OpenSimplex2");
    Py_INCREF(self->domainWarpType);
    self->fractalType = PyUnicode_FromString("None");
    Py_INCREF(self->fractalType);
    self->rotationType3d = PyUnicode_FromString("None");
    Py_INCREF(self->rotationType3d);
    self->octaves = Py_BuildValue("i", 3);
    Py_INCREF(self->octaves);
    self->lacunarity = PyFloat_FromDouble(2.0);
    Py_INCREF(self->lacunarity);
    self->gain = PyFloat_FromDouble(0.5);
    Py_INCREF(self->gain);
    self->weightedStrength = PyFloat_FromDouble(0.0);
    Py_INCREF(self->weightedStrength);
    self->pingPongStrength = PyFloat_FromDouble(2.0);
    Py_INCREF(self->pingPongStrength);
    self->cellularJitter = PyFloat_FromDouble(1.0);
    Py_INCREF(self->cellularJitter);
    self->domainWarpAmp = PyFloat_FromDouble(1.0);
    Py_INCREF(self->domainWarpAmp);
    return 0;
}

static PyObject* Noise_set_frequency(NoiseObject* self, PyObject* args) {
    float frequency;
    if (!PyArg_ParseTuple(args, "f", &frequency)) {
        PyErr_SetString(PyExc_TypeError, "Input must be a number");
        return NULL;
    }
    self->noise->SetFrequency(frequency);
    Py_RETURN_NONE;
}

static PyObject* Noise_get_image(NoiseObject* self, PyObject* args) {
    int resX, resY;
    if (!PyArg_ParseTuple(args, "ii", &resX, &resY)) {
        PyErr_SetString(PyExc_TypeError, "Input must be two integers");
        return NULL;
    }
    const npy_intp dims[] = {resX, resY};
    int size = resX*resY;
    npy_float data[size];
    int ind = 0;
    for (int i = 0; i < resX; i++) {
        for (int j = 0; j < resY; j++) {
            double noiseVal = self->noise->GetNoise((float)i, (float)j);
            data[ind++] = ((noiseVal+1)/2)*255;
        }
    }
    PyObject* arr = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, data);
    PyObject* image = PyImport_ImportModule("PIL.Image");
    PyObject* fromarr = PyObject_GetAttrString(image, "fromarray");
    PyObject* arrTup = Py_BuildValue("(O)", arr);
    PyObject* im = PyObject_CallObject(fromarr, arrTup);
    Py_DECREF(image);
    Py_DECREF(arr);
    Py_DECREF(arrTup);
    Py_DECREF(fromarr);
    PyObject* lType = PyUnicode_FromString("L");
    PyObject* convert = PyObject_GetAttrString(im, "convert");
    PyObject* lTup = Py_BuildValue("(O)", lType);
    PyObject* tmp = PyObject_CallObject(convert, lTup);
    Py_DECREF(im);
    Py_DECREF(lTup);
    im = tmp;
    Py_DECREF(lType);
    Py_DECREF(convert);
    return im;
}

static PyObject* Noise_get_noise(NoiseObject* self, PyObject* args) {
    float x, y, z;
    if (PyTuple_Size(args) == 3) {
        if (!PyArg_ParseTuple(args, "fff", &x, &y, &z)) {
            PyErr_SetString(PyExc_TypeError, "Invalid input");
            return NULL;
        }
        float noise = self->noise->GetNoise(x, y, z);
        return PyFloat_FromDouble((double)noise);
    } else if (PyTuple_Size(args) == 2) {
        if (!PyArg_ParseTuple(args, "ff", &x, &y)) {
            PyErr_SetString(PyExc_TypeError, "Invalid input");
            return NULL;
        }
        float noise = self->noise->GetNoise(x, y);
        return PyFloat_FromDouble((double)noise);
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid input");
        return NULL;
    }
}

static PyObject* Noise_set_seed(NoiseObject* self, PyObject* args) {
    int seed;
    if (!PyArg_ParseTuple(args, "i", &seed)) {
        PyErr_SetString(PyExc_TypeError, "Input must be an integer");
        return NULL;
    }
    self->noise->SetSeed(seed);
    Py_RETURN_NONE;
}

static PyObject* Noise_set_noise_type(NoiseObject* self, PyObject* args) {
    char* input;
    if (!PyArg_ParseTuple(args, "s", &input)) {
        PyErr_SetString(PyExc_TypeError, "Input must be a string");
        return NULL;
    }
    if (!strcmp(input, "OpenSimplex2")) {
        self->noise->SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
        Py_DECREF(self->noiseType);
        self->noiseType = PyUnicode_FromString("OpenSimplex2");
        Py_INCREF(self->noiseType);
    } else if (!strcmp(input, "OpenSimplex2S")) {
        self->noise->SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2S);
        Py_DECREF(self->noiseType);
        self->noiseType = PyUnicode_FromString("OpenSimplex2S");
        Py_INCREF(self->noiseType);
    } else if (!strcmp(input, "Cellular")) {
        self->noise->SetNoiseType(FastNoiseLite::NoiseType_Cellular);
        Py_DECREF(self->noiseType);
        self->noiseType = PyUnicode_FromString("Cellular");
        Py_INCREF(self->noiseType);
    } else if (!strcmp(input, "Perlin")) {
        self->noise->SetNoiseType(FastNoiseLite::NoiseType_Perlin);
        Py_DECREF(self->noiseType);
        self->noiseType = PyUnicode_FromString("Perlin");
        Py_INCREF(self->noiseType);
    } else if (!strcmp(input, "ValueCubic")) {
        self->noise->SetNoiseType(FastNoiseLite::NoiseType_ValueCubic);
        Py_DECREF(self->noiseType);
        self->noiseType = PyUnicode_FromString("ValueCubic");
        Py_INCREF(self->noiseType);
    } else if (!strcmp(input, "Value")) {
        self->noise->SetNoiseType(FastNoiseLite::NoiseType_Value);
        Py_DECREF(self->noiseType);
        self->noiseType = PyUnicode_FromString("Value");
        Py_INCREF(self->noiseType);
    } else {
        PyErr_SetString(PyExc_AssertionError, "Unrecognized noise type");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* Noise_set_fractal_type(NoiseObject* self, PyObject* args) {
    char* input;
    if (!PyArg_ParseTuple(args, "s", &input)) {
        PyErr_SetString(PyExc_TypeError, "Input must be a string");
        return NULL;
    }
    if (!strcmp(input, "None")) {
        self->noise->SetFractalType(FastNoiseLite::FractalType_None);
        Py_DECREF(self->fractalType);
        self->fractalType = PyUnicode_FromString("None");
        Py_INCREF(self->fractalType);
    } else if (!strcmp(input, "FBm")) {
        self->noise->SetFractalType(FastNoiseLite::FractalType_FBm);
        Py_DECREF(self->fractalType);
        self->fractalType = PyUnicode_FromString("FBm");
        Py_INCREF(self->fractalType);
    } else if (!strcmp(input, "Ridged")) {
        self->noise->SetFractalType(FastNoiseLite::FractalType_Ridged);
        Py_DECREF(self->fractalType);
        self->fractalType = PyUnicode_FromString("Ridged");
        Py_INCREF(self->fractalType);
    } else if (!strcmp(input, "PingPong")) {
        self->noise->SetFractalType(FastNoiseLite::FractalType_PingPong);
        Py_DECREF(self->fractalType);
        self->fractalType = PyUnicode_FromString("PingPong");
        Py_INCREF(self->fractalType);
    } else if (!strcmp(input, "DomainWarpProgressive")) {
        self->noise->SetFractalType(FastNoiseLite::FractalType_DomainWarpProgressive);
        Py_DECREF(self->fractalType);
        self->fractalType = PyUnicode_FromString("DomainWarpProgressive");
        Py_INCREF(self->fractalType);
    } else if (!strcmp(input, "DomainWarpIndependent")) {
        self->noise->SetFractalType(FastNoiseLite::FractalType_DomainWarpIndependent);
        Py_DECREF(self->fractalType);
        self->fractalType = PyUnicode_FromString("DomainWarpIndependent");
        Py_INCREF(self->fractalType);
    } else {
        PyErr_SetString(PyExc_AssertionError, "Unrecognized fractal type");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* Noise_set_fractal_octaves(NoiseObject* self, PyObject* args) {
    int octaves;
    if (!PyArg_ParseTuple(args, "i", &octaves)) {
        PyErr_SetString(PyExc_TypeError, "Input must be an integer");
        return NULL;
    }
    self->noise->SetFractalOctaves(octaves);
    Py_DECREF(self->octaves);
    self->octaves = Py_BuildValue("i", octaves);
    Py_INCREF(self->octaves);
    Py_RETURN_NONE;
}

static PyObject* Noise_set_fractal_lacunarity(NoiseObject* self, PyObject* args) {
    float lacunarity;
    if (!PyArg_ParseTuple(args, "f", &lacunarity)) {
        PyErr_SetString(PyExc_TypeError, "Input must be a float");
        return NULL;
    }
    self->noise->SetFractalLacunarity(lacunarity);
    Py_DECREF(self->lacunarity);
    self->lacunarity = PyFloat_FromDouble((double)lacunarity);
    Py_INCREF(self->lacunarity);
    Py_RETURN_NONE;
}

static PyObject* Noise_set_fractal_gain(NoiseObject* self, PyObject* args) {
    float gain;
    if (!PyArg_ParseTuple(args, "f", &gain)) {
        PyErr_SetString(PyExc_TypeError, "Input must be a float");
        return NULL;
    }
    self->noise->SetFractalGain(gain);
    Py_DECREF(self->gain);
    self->gain = PyFloat_FromDouble((double)gain);
    Py_INCREF(self->gain);
    Py_RETURN_NONE;
}

static PyObject* Noise_set_fractal_weighted_strength(NoiseObject* self, PyObject* args) {
    float weightedStrength;
    if (!PyArg_ParseTuple(args, "f", &weightedStrength)) {
        PyErr_SetString(PyExc_TypeError, "Input must be a float");
        return NULL;
    }
    self->noise->SetFractalWeightedStrength(weightedStrength);
    Py_DECREF(self->weightedStrength);
    self->weightedStrength = PyFloat_FromDouble((double)weightedStrength);
    Py_INCREF(self->weightedStrength);
    Py_RETURN_NONE;
}

static PyObject* Noise_set_fractal_ping_pong_strength(NoiseObject* self, PyObject* args) {
    float pingPongStrength;
    if (!PyArg_ParseTuple(args, "f", &pingPongStrength)) {
        PyErr_SetString(PyExc_TypeError, "Input must be a float");
        return NULL;
    }
    self->noise->SetFractalPingPongStrength(pingPongStrength);
    Py_DECREF(self->pingPongStrength);
    self->pingPongStrength = PyFloat_FromDouble((double)pingPongStrength);
    Py_INCREF(self->pingPongStrength);
    Py_RETURN_NONE;
}

static PyObject* Noise_domain_warp(NoiseObject* self, PyObject* args) {
    float x, y, z;
    if (PyTuple_Size(args) == 3) {
        if (!PyArg_ParseTuple(args, "fff", &x, &y, &z)) {
            PyErr_SetString(PyExc_TypeError, "Invalid input");
            return NULL;
        }
        self->noise->DomainWarp(x, y, z);
    } else if (PyTuple_Size(args) == 2) {
        if (!PyArg_ParseTuple(args, "ff", &x, &y)) {
            PyErr_SetString(PyExc_TypeError, "Invalid input");
            return NULL;
        }
        self->noise->DomainWarp(x, y);
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid input");
        return NULL;
    }
    Py_RETURN_NONE;
}

PyObject* Noise_get_noise_2d(NoiseObject* self, PyObject* args, PyObject *kwargs) {
    int x, y;
    double xStep = 1, yStep = 1;
    PyObject* stepObj;
    static const char* kws[] = {"x", "y", "step", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii|O", const_cast<char **>(kws), &x, &y, &stepObj)) {
        PyErr_SetString(PyExc_TypeError, "Input must be two integers and optional step");
        return NULL;
    }
    if (PyObject_TypeCheck(stepObj, &PyTuple_Type)) {
        if (!PyArg_ParseTuple(stepObj, "dd", &xStep, &yStep)) {
            PyErr_SetString(PyExc_TypeError, "Step must be a number of a tuple of two numbers");
            return NULL;
        }
    } else {
        PyObject* tup = PyTuple_Pack(1, stepObj);
        double step;
        if (!PyArg_ParseTuple(tup, "d", &step)) {
            PyErr_SetString(PyExc_TypeError, "Step must be a number of a tuple of two numbers");
            return NULL;
        }
        xStep = step;
        yStep = step;
    }
    npy_intp dims[] = {x, y};
    PyObject* out = PyArray_ZEROS(2, dims, NPY_FLOAT64, 0);
    auto data = (npy_float64*)PyArray_DATA((PyArrayObject*)out);
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            data[i*y+j] = self->noise->GetNoise((float)(((double)i)/xStep), (float)(((double)j)/yStep));
        }
    }
    return out;
}

PyObject* Noise_get_noise_3d(NoiseObject* self, PyObject* args, PyObject *kwargs) {
    int x, y, z;
    double xStep = 1, yStep = 1, zStep = 1;
    PyObject* stepObj;
    static const char* kws[] = {(char*)"x", "y", "z", "step", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iii|O", const_cast<char **>(kws), &x, &y, &z, &stepObj)) {
        PyErr_SetString(PyExc_TypeError, "Input must be three integers and optional step");
        return NULL;
    }
    if (PyObject_TypeCheck(stepObj, &PyTuple_Type)) {
        if (!PyArg_ParseTuple(stepObj, "ddd", &xStep, &yStep)) {
            PyErr_SetString(PyExc_TypeError, "Step must be a number of a tuple of three numbers");
            return NULL;
        }
    } else {
        PyObject* tup = PyTuple_Pack(1, stepObj);
        double step;
        if (!PyArg_ParseTuple(tup, "d", &step)) {
            PyErr_SetString(PyExc_TypeError, "Step must be a number of a tuple of three numbers");
            return NULL;
        }
        xStep = step;
        yStep = step;
        zStep = step;
    }
    npy_intp dims[] = {x, y, z};
    PyObject* out = PyArray_ZEROS(3, dims, NPY_FLOAT64, 0);
    auto data = (npy_float64*)PyArray_DATA((PyArrayObject*)out);
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            for (int k = 0; k < z; k++) {
                data[i*y*z+j*z+k] = self->noise->GetNoise((float)(((double)i)/xStep), (float)(((double)j)/yStep), (float)(((double)j)/zStep));
            }
        }
    }
    return out;
}

static PyObject* Noise_set_cellular_distance_function(NoiseObject* self, PyObject* args) {
    char* input;
    if (!PyArg_ParseTuple(args, "s", &input)) {
        PyErr_SetString(PyExc_TypeError, "Input must be a string");
        return NULL;
    }
    if (!strcmp(input, "Euclidean")) {
        self->noise->SetCellularDistanceFunction(FastNoiseLite::CellularDistanceFunction_Euclidean);
        Py_DECREF(self->cellularDistanceFunction);
        self->cellularDistanceFunction = PyUnicode_FromString("Euclidean");
        Py_INCREF(self->cellularDistanceFunction);
    } else if (!strcmp(input, "EuclideanSq")) {
        self->noise->SetCellularDistanceFunction(FastNoiseLite::CellularDistanceFunction_EuclideanSq);
        Py_DECREF(self->cellularDistanceFunction);
        self->cellularDistanceFunction = PyUnicode_FromString("EuclideanSq");
        Py_INCREF(self->cellularDistanceFunction);
    } else if (!strcmp(input, "Manhattan")) {
        self->noise->SetCellularDistanceFunction(FastNoiseLite::CellularDistanceFunction_Manhattan);
        Py_DECREF(self->cellularDistanceFunction);
        self->cellularDistanceFunction = PyUnicode_FromString("Manhattan");
        Py_INCREF(self->cellularDistanceFunction);
    } else if (!strcmp(input, "Hybrid")) {
        self->noise->SetCellularDistanceFunction(FastNoiseLite::CellularDistanceFunction_Hybrid);
        Py_DECREF(self->cellularDistanceFunction);
        self->cellularDistanceFunction = PyUnicode_FromString("Hybrid");
        Py_INCREF(self->cellularDistanceFunction);
    } else {
        PyErr_SetString(PyExc_AssertionError, "Unrecognized cellular distance function. Valid functions are: Euclidean, EuclideanSq, Manhattan, Hybrid");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* Noise_set_cellular_return_type(NoiseObject* self, PyObject* args) {
    char* input;
    if (!PyArg_ParseTuple(args, "s", &input)) {
        PyErr_SetString(PyExc_TypeError, "Input must be a string");
        return NULL;
    }
    if (!strcmp(input, "CellValue")) {
        self->noise->SetCellularReturnType(FastNoiseLite::CellularReturnType_CellValue);
        Py_DECREF(self->cellularReturnType);
        self->cellularReturnType = PyUnicode_FromString("CellValue");
        Py_INCREF(self->cellularReturnType);
    } else if (!strcmp(input, "Distance")) {
        self->noise->SetCellularReturnType(FastNoiseLite::CellularReturnType_Distance);
        Py_DECREF(self->cellularReturnType);
        self->cellularReturnType = PyUnicode_FromString("Distance");
        Py_INCREF(self->cellularReturnType);
    } else if (!strcmp(input, "Distance2")) {
        self->noise->SetCellularReturnType(FastNoiseLite::CellularReturnType_Distance2);
        Py_DECREF(self->cellularReturnType);
        self->cellularReturnType = PyUnicode_FromString("Distance2");
        Py_INCREF(self->cellularReturnType);
    } else if (!strcmp(input, "Distance2Add")) {
        self->noise->SetCellularReturnType(FastNoiseLite::CellularReturnType_Distance2Add);
        Py_DECREF(self->cellularReturnType);
        self->cellularReturnType = PyUnicode_FromString("Distance2Add");
        Py_INCREF(self->cellularReturnType);
    } else if (!strcmp(input, "Distance2Sub")) {
        self->noise->SetCellularReturnType(FastNoiseLite::CellularReturnType_Distance2Sub);
        Py_DECREF(self->cellularReturnType);
        self->cellularReturnType = PyUnicode_FromString("Distance2Sub");
        Py_INCREF(self->cellularReturnType);
    } else if (!strcmp(input, "Distance2Mul")) {
        self->noise->SetCellularReturnType(FastNoiseLite::CellularReturnType_Distance2Mul);
        Py_DECREF(self->cellularReturnType);
        self->cellularReturnType = PyUnicode_FromString("Distance2Mul");
        Py_INCREF(self->cellularReturnType);
    } else if (!strcmp(input, "Distance2Div")) {
        self->noise->SetCellularReturnType(FastNoiseLite::CellularReturnType_Distance2Div);
        Py_DECREF(self->cellularReturnType);
        self->cellularReturnType = PyUnicode_FromString("Distance2Div");
        Py_INCREF(self->cellularReturnType);
    } else {
        PyErr_SetString(PyExc_AssertionError, "Unrecognized cellular return type. Valid types are: CellValue, Distance, Distance2, Distance2Add, Distance2Sub, Distance2Mul, Distance2Div");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* Noise_set_cellular_jitter(NoiseObject* self, PyObject* args) {
    float cellularJitter;
    if (!PyArg_ParseTuple(args, "f", &cellularJitter)) {
        PyErr_SetString(PyExc_TypeError, "Input must be a number");
        return NULL;
    }
    self->noise->SetCellularJitter(cellularJitter);
    Py_DECREF(self->cellularJitter);
    self->cellularJitter = Py_BuildValue("f", cellularJitter);
    Py_INCREF(self->cellularJitter);
    Py_RETURN_NONE;
}

static PyObject* Noise_set_domain_warp_type(NoiseObject* self, PyObject* args) {
    char* input;
    if (!PyArg_ParseTuple(args, "s", &input)) {
        PyErr_SetString(PyExc_TypeError, "Input must be a string");
        return NULL;
    }
    if (!strcmp(input, "OpenSimplex2")) {
        self->noise->SetDomainWarpType(FastNoiseLite::DomainWarpType_OpenSimplex2);
        Py_DECREF(self->domainWarpType);
        self->domainWarpType = PyUnicode_FromString("OpenSimplex2");
        Py_INCREF(self->domainWarpType);
    } else if (!strcmp(input, "OpenSimplex2Reduced")) {
        self->noise->SetDomainWarpType(FastNoiseLite::DomainWarpType_OpenSimplex2Reduced);
        Py_DECREF(self->domainWarpType);
        self->domainWarpType = PyUnicode_FromString("OpenSimplex2Reduced");
        Py_INCREF(self->domainWarpType);
    } else if (!strcmp(input, "BasicGrid")) {
        self->noise->SetDomainWarpType(FastNoiseLite::DomainWarpType_BasicGrid);
        Py_DECREF(self->domainWarpType);
        self->domainWarpType = PyUnicode_FromString("BasicGrid");
        Py_INCREF(self->domainWarpType);
    } else {
        PyErr_SetString(PyExc_AssertionError, "Unrecognized domain warp type. Valid types are: OpenSimplex2, OpenSimplex2Reduced, BasicGrid");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* Noise_set_domain_warp_amp(NoiseObject* self, PyObject* args) {
    float domainWarpAmp;
    if (!PyArg_ParseTuple(args, "f", &domainWarpAmp)) {
        PyErr_SetString(PyExc_TypeError, "Input must be a number");
        return NULL;
    }
    self->noise->SetDomainWarpAmp(domainWarpAmp);
    Py_DECREF(self->domainWarpAmp);
    self->domainWarpAmp = Py_BuildValue("f", domainWarpAmp);
    Py_INCREF(self->domainWarpAmp);
    Py_RETURN_NONE;
}

static PyObject* Noise_set_rotation_type_3d(NoiseObject* self, PyObject* args) {
    char* input;
    if (!PyArg_ParseTuple(args, "s", &input)) {
        PyErr_SetString(PyExc_TypeError, "Input must be a string");
        return NULL;
    }
    if (!strcmp(input, "None")) {
        self->noise->SetRotationType3D(FastNoiseLite::RotationType3D_None);
        Py_DECREF(self->rotationType3d);
        self->rotationType3d = PyUnicode_FromString("None");
        Py_INCREF(self->rotationType3d);
    } else if (!strcmp(input, "ImproveXYPlanes")) {
        self->noise->SetRotationType3D(FastNoiseLite::RotationType3D_ImproveXYPlanes);
        Py_DECREF(self->rotationType3d);
        self->rotationType3d = PyUnicode_FromString("ImproveXYPlanes");
        Py_INCREF(self->rotationType3d);
    } else if (!strcmp(input, "ImproveXZPlanes")) {
        self->noise->SetRotationType3D(FastNoiseLite::RotationType3D_ImproveXZPlanes);
        Py_DECREF(self->rotationType3d);
        self->rotationType3d = PyUnicode_FromString("ImproveXZPlanes");
        Py_INCREF(self->rotationType3d);
    } else {
        PyErr_SetString(PyExc_AssertionError, "Unrecognized rotation type. Recognized types are: None, ImproveXYPlanes, ImproveXZPlanes");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyMethodDef Noise_methods[] = {
        {"set_frequency", (PyCFunction) Noise_set_frequency, METH_VARARGS, "Sets the noise frequency"},
        {"get_image", (PyCFunction) Noise_get_image, METH_VARARGS, "Gets an image of the specified size"},
        {"get_noise", (PyCFunction) Noise_get_noise, METH_VARARGS, "Gets the noise at the specified location"},
        {"set_seed", (PyCFunction) Noise_set_seed, METH_VARARGS, "Sets the seed of the noise"},
        {"set_noise_type", (PyCFunction) Noise_set_noise_type, METH_VARARGS, "Sets the noise type"},
        {"set_fractal_type", (PyCFunction) Noise_set_fractal_type, METH_VARARGS, "Sets the fractal type"},
        {"set_fractal_octaves", (PyCFunction) Noise_set_fractal_octaves, METH_VARARGS, "Sets the fractal octaves"},
        {"set_fractal_lacunarity", (PyCFunction) Noise_set_fractal_lacunarity, METH_VARARGS, "Sets the fractal lacunarity"},
        {"set_fractal_gain", (PyCFunction) Noise_set_fractal_gain, METH_VARARGS, "Sets the fractal gain"},
        {"set_fractal_weighted_strength", (PyCFunction) Noise_set_fractal_weighted_strength, METH_VARARGS, "Sets the fractal weighted strength"},
        {"set_fractal_ping_pong_strength", (PyCFunction) Noise_set_fractal_ping_pong_strength, METH_VARARGS, "Sets the fractal ping pong strength"},
        {"domain_warp", (PyCFunction) Noise_domain_warp, METH_VARARGS, "Warps the domain"},
        {"get_noise_2d", (PyCFunction) Noise_get_noise_2d, METH_VARARGS | METH_KEYWORDS, "Gets a 2d numpy array of the specified size of noise"},
        {"get_noise_3d", (PyCFunction) Noise_get_noise_3d, METH_VARARGS, "Gets a 3d numpy array of the specified size of noise"},
        {"set_cellular_distance_function", (PyCFunction) Noise_set_cellular_distance_function, METH_VARARGS, "Sets the cellular distance function"},
        {"set_cellular_return_type", (PyCFunction) Noise_set_cellular_return_type, METH_VARARGS, "Sets the cellular return type"},
        {"set_cellular_jitter", (PyCFunction) Noise_set_cellular_jitter, METH_VARARGS, "Sets the cellular jitter"},
        {"set_domain_warp_type", (PyCFunction) Noise_set_domain_warp_type, METH_VARARGS, "Sets the domain warp type"},
        {"set_domain_warp_amp", (PyCFunction) Noise_set_domain_warp_amp, METH_VARARGS, "Sets the domain warp amp"},
        {"set_rotation_type_3d", (PyCFunction) Noise_set_rotation_type_3d, METH_VARARGS, "Sets the rotation type"},
        {NULL}
};

static PyMemberDef Noise_members[] = {
        {"noise_type", T_OBJECT_EX, offsetof(NoiseObject, noiseType), 0,
                "The noise type"},
        {"fractal_type", T_OBJECT_EX, offsetof(NoiseObject, fractalType), 0, "The fractal type"},
        {"fractal_octaves", T_OBJECT_EX, offsetof(NoiseObject, octaves), 0, "The fractal octaves"},
        {"fractal_lacunarity", T_OBJECT_EX, offsetof(NoiseObject, lacunarity), 0, "The fractal lacunarity"},
        {"fractal_gain", T_OBJECT_EX, offsetof(NoiseObject, gain), 0, "The fractal gain"},
        {"fractal_weighted_strength", T_OBJECT_EX, offsetof(NoiseObject, weightedStrength), 0, "The fractal weighted strength"},
        {"fractal_ping_pong_strength", T_OBJECT_EX, offsetof(NoiseObject, pingPongStrength), 0, "The fractal ping pong strength"},
        {"cellular_distance_function", T_OBJECT_EX, offsetof(NoiseObject, cellularDistanceFunction), 0, "The cellular distance function"},
        {"cellular_return_type", T_OBJECT_EX, offsetof(NoiseObject, cellularReturnType), 0, "The cellular return type"},
        {"cellular_jitter", T_OBJECT_EX, offsetof(NoiseObject, cellularJitter), 0, "The cellular jitter"},
        {"domain_warp_type", T_OBJECT_EX, offsetof(NoiseObject, domainWarpType), 0, "The domain warp type"},
        {"domain_warp_amp", T_OBJECT_EX, offsetof(NoiseObject, domainWarpAmp), 0, "The domain warp amp"},
        {"rotation_type_3d", T_OBJECT_EX, offsetof(NoiseObject, rotationType3d), 0, "The rotation type"},

        {NULL}
};

static PyTypeObject NoiseType = {
        NULL,
        0,
        0,
        "noise.Noise",
        sizeof(NoiseObject),
        0,
        (destructor) Noise_dealloc,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        Py_TPFLAGS_DEFAULT,
        PyDoc_STR("Noise object"),
        0,0,0,0,0,0,
        Noise_methods,
        Noise_members,
        0,0,0,0,0,0,
        (initproc)Noise_init,
        0,
        PyType_GenericNew,
        0,0,0,0,0,0,0,0,0,0,0
};

static PyModuleDef noise_module = {
        PyModuleDef_HEAD_INIT,
        "noise",
        "FastNoiseLite Python library",
        -1,
};

PyMODINIT_FUNC PyInit_noise(void) {
    PyObject *m;
    if (PyType_Ready(&NoiseType) < 0) {
        return NULL;
    }
    m = PyModule_Create(&noise_module);
    if (m == NULL) {
        return NULL;
    }
    Py_INCREF(&NoiseType);
    if (PyModule_AddObject(m, "Noise", (PyObject*) &NoiseType) < 0) {
        return NULL;
    }
    import_array();
    return m;
}