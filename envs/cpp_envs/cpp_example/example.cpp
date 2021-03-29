#include <pybind11/pybind11.h>

namespace py = pybind11;

// Small addition example from pybind11
int add(int i, int j) {
    return i + j;
}

// TODO: add binding code to separate file
PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");
}
