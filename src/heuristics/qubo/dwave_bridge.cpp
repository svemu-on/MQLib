#include "heuristics/qubo/dwave_bridge.h"

#ifdef USE_DWAVE

#include <iostream>
#include <stdexcept>
#include <string>

#include <pybind11/embed.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Small helper to ensure the Python interpreter is initialised exactly
// once per process.
namespace {
struct PythonEnvironment {
    PythonEnvironment() : guard() {
        // nothing else to do
    }
    py::scoped_interpreter guard;
};

PythonEnvironment& get_env() {
    static PythonEnvironment env;
    return env;
}
}  // namespace

DWaveResult run_dwave_solver(
    const std::vector<std::tuple<int,int,double>>& qubo_terms,
    const std::string& backend,
    const std::string& config_json_path,
    std::string* error_message)
{
    (void)get_env();  // ensure interpreter is initialised

    DWaveResult result;
    if (error_message) {
        error_message->clear();
    }

    try {
        py::gil_scoped_acquire gil;

        // Make sure our local python/ directory is on sys.path
        py::module_ sys = py::module_::import("sys");
        py::list sys_path = sys.attr("path");
        
        // 1) Add our local python/ helper directory (for mqlib_dwave.py)
        sys_path.insert(0, "python");

        // 2) Try to add the project-local .venv site-packages so that
        //    dimod / dwave-ocean-sdk installed there are visible.
        //
        //    We assume the process is started from the MQLib repo root
        //    and that a uv venv exists at .venv created with:
        //        uv venv .venv
        //
        try {
            py::module_ os = py::module_::import("os");
            py::object version_info = sys.attr("version_info");
            int major = version_info.attr("major").cast<int>();
            int minor = version_info.attr("minor").cast<int>();

            // Construct ".venv/lib/pythonX.Y/site-packages"
            std::string venv_site =
                std::string(".venv/lib/python") +
                std::to_string(major) + "." +
                std::to_string(minor) + "/site-packages";

            py::object isdir = os.attr("path").attr("isdir");
            if (isdir(py::str(venv_site)).cast<bool>()) {
                sys_path.insert(0, venv_site);
            }
        } catch (const std::exception&) {
            // Ignore failures here; we just fall back to whatever sys.path already has.
        }

        py::module_ m = py::module_::import("mqlib_dwave");
        py::object solve_qubo = m.attr("solve_qubo");

        py::list terms_py;
        for (const auto& t : qubo_terms) {
            int i, j;
            double w;
            std::tie(i, j, w) = t;
            terms_py.append(py::make_tuple(i, j, w));
        }

        py::object cfg_path_obj;
        if (config_json_path.empty()) {
            cfg_path_obj = py::none();
        } else {
            cfg_path_obj = py::str(config_json_path);
        }

        // solve_qubo returns (assignments: List[int], weight: float)
        py::object res = solve_qubo(terms_py, backend, cfg_path_obj);
        py::sequence seq = res;

        py::object sample_obj = seq[0];
        py::object weight_obj = seq[1];

        result.best_sample = sample_obj.cast<std::vector<int>>();
        result.best_weight = weight_obj.cast<double>();
    } catch (const std::exception& e) {
        if (error_message) {
            *error_message = e.what();
        }
        result.best_sample.clear();
        result.best_weight = 0.0;
    } catch (...) {
        if (error_message) {
            *error_message = "Unknown exception in run_dwave_solver";
        }
        result.best_sample.clear();
        result.best_weight = 0.0;
    }

    return result;
}

#else  // !USE_DWAVE

DWaveResult run_dwave_solver(
    const std::vector<std::tuple<int,int,double>>&,
    const std::string&,
    const std::string&,
    std::string* error_message)
{
    if (error_message) {
        *error_message = "D-Wave support not compiled (USE_DWAVE not defined)";
    }
    DWaveResult result;
    result.best_sample.clear();
    result.best_weight = 0.0;
    return result;
}

#endif  // USE_DWAVE
