"""Make the venv's bundled CUDA runtime libraries discoverable BEFORE
onnxruntime / torch try to load them.

onnxruntime-gpu (and torch's cu13 wheels) ship the CUDA runtime as separate
`nvidia-*-cu13` packages inside the virtual-env's site-packages. Those library
directories are not on the OS loader path by default, so the GPU providers fail
to load unless we register them here.

Cross-platform behaviour:
  * Linux/macOS: the C loader reads ``LD_LIBRARY_PATH`` / ``DYLD_LIBRARY_PATH``
    only at process start, so we prepend the ``nvidia/cu13/lib`` dir (which holds
    ``libcudart.so.13`` etc.) and re-exec the interpreter once. Guarded against a
    re-exec loop: on the second run the dir is already present.
  * Windows: ``os.add_dll_directory()`` affects the DLL search in-process, so we
    register the CUDA DLL directories (nvidia + torch) with no re-exec.

Library locations are derived from ``sysconfig`` (the running venv's
site-packages) — nothing is hard-coded to a platform, distro, or python version.

NOTE: the Windows path is untested on real Windows hardware (developed on Linux).
"""

import os
import sys
import glob
import sysconfig


def _site_roots():
    roots = []
    try:
        roots.append(sysconfig.get_paths()["purelib"])   # venv site-packages
    except Exception:
        pass
    roots += [p for p in getattr(sys, "path", []) if p.endswith("site-packages")]
    out, seen = [], set()
    for r in roots:
        if r and os.path.isdir(r) and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _linux_cuda_libdir():
    """The dir onnxruntime needs on LD_LIBRARY_PATH (libcudart.so.13 etc.)."""
    for root in _site_roots():
        d = os.path.join(root, "nvidia", "cu13", "lib")
        if os.path.isdir(d):
            return os.path.normpath(d)
    return None


def _windows_dll_dirs():
    """All dirs that may hold CUDA/cuDNN DLLs for onnxruntime + torch on Windows."""
    dirs = []
    for root in _site_roots():
        for sub in ("bin", "lib"):
            for cand in (os.path.join(root, "nvidia", "cu13", sub),
                         os.path.join(root, "torch", sub)):
                if os.path.isdir(cand):
                    dirs.append(cand)
            dirs += [d for d in glob.glob(os.path.join(root, "nvidia", "*", sub))
                     if os.path.isdir(d)]
    out, seen = [], set()
    for d in dirs:
        d = os.path.normpath(d)
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def ensure_cuda_libpath():
    if sys.platform == "win32":
        for d in _windows_dll_dirs():
            try:
                os.add_dll_directory(d)          # in-process; Python 3.8+
            except (OSError, AttributeError):
                pass
        return

    libdir = _linux_cuda_libdir()
    if not libdir:
        return
    env_var = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    current = os.environ.get(env_var, "")
    if libdir not in current.split(os.pathsep):
        os.environ[env_var] = libdir + os.pathsep + current
        # The loader already read the path at start; re-exec so it takes effect.
        os.execv(sys.executable, [sys.executable] + sys.argv)
