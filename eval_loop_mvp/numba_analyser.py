import inspect
import site
import types as pytypes
import os
from collections import namedtuple
from contextlib import contextmanager

import pandas as pd

from pixie.eval_hook import create_custom_hook

from numba import njit, types, typeof
from numba.core import ir
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.untyped_passes import IRProcessing

site_pkgs = os.path.dirname(site.getsitepackages()[-1])

_log = []
log_entry = namedtuple('log_entry', 'event function argtys status meta')


class Cache():
    def __init__(self):
        self._compile_cache = dict()

    # a cache impl, not using functools as this can be more easily debugged
    def add(self, fn):
        jitted = rjit(fn)
        if fn in self._compile_cache:
            return self._compile_cache[fn]
        else:
            self._compile_cache[fn] = jitted
            return jitted


_compile_cache = Cache()


@register_pass(mutates_CFG=True, analysis_only=False)
class JitAllFunctions(FunctionPass):
    _name = "jit_all_functions"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        func_ir = state.func_ir
        mutated = False
        swaps = dict()
        for blk in func_ir.blocks.values():
            for assgn in blk.find_insts(ir.Assign):
                assgnval = assgn.value
                if isinstance(assgnval, ir.Expr) and assgnval.op == 'call':
                    callglbl = func_ir.get_definition(assgn.value.func)
                    if isinstance(callglbl, ir.Global):
                        pyfunc = callglbl.value
                        if isinstance(pyfunc, pytypes.FunctionType):
                            jitwrap = _compile_cache.add(pyfunc)
                            # new global
                            newglbl = ir.Global(callglbl.name, jitwrap,
                                                callglbl.loc)
                            lhs = func_ir.get_assignee(callglbl)
                            swaps[(lhs, callglbl)] = newglbl
                            mutated |= True
        # do the swaps
        for blk in func_ir.blocks.values():
            for assgn in blk.find_insts(ir.Assign):
                swap_to = swaps.get((assgn.target, assgn.value))
                if swap_to is not None:
                    assgn.value = swap_to
        return mutated


class RecursiveJitCompiler(CompilerBase):

    def define_pipelines(self):
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
        pm.add_pass_after(JitAllFunctions, IRProcessing)
        pm.finalize()
        return [pm]


rjit = njit(pipeline_class=RecursiveJitCompiler)


def numba_jit_analyser(frame_stuff, co_extra):

    # per code-object execution counter
    if 'visited' in co_extra:
        co_extra['visited'] += 1
    else:
        co_extra['visited'] = 0

    fn = frame_stuff['f_func']
    stack_args = frame_stuff['localsplus']

    try:
        argtys = tuple([typeof(x) for x in stack_args])

        fname = inspect.getfile(fn)

        if site_pkgs in fname or fname.startswith('<frozen '):
            # skip site packages and frozen functions, can't JIT these
            entry = log_entry(event='jit',
                              function=fn,
                              argtys=argtys,
                              status="skip",
                              meta=None)
            _log.append(entry)
            return (False, None)
        else:
            # add info about the current function
            entry = log_entry(event='info',
                              function=fn,
                              argtys=argtys,
                              status=None,
                              meta=None)
            _log.append(entry)
        try:
            # try cache first
            jitted = _compile_cache.add(fn)
            tmp = jitted(*stack_args)
            entry = log_entry(event='jit',
                              function=fn,
                              argtys=argtys,
                              status="success",
                              meta=None)
            _log.append(entry)
            return (True, tmp)
        except Exception as e:
            # JIT failed, add log entry with exception
            entry = log_entry(event='jit',
                              function=fn,
                              argtys=argtys,
                              status="failure",
                              meta={'exception': e})
            _log.append(entry)
    except Exception as e:
        # something unexpected happened, log what it was
        entry = log_entry(event='unhandled_error',
                          function=fn,
                          argtys=None,
                          status="failure",
                          meta={'exception': e})
        _log.append(entry)
    return (False, None)


class LogHandler():
    def __init__(self, log):
        self._log = log

    def getlog(self):
        return self._log[:]

    def process(self):
        print("")
        print("__ Analysis __\n")
        cp = self.getlog()
        df = pd.DataFrame(cp, index=range(len(cp)))
        # add in canonical mod
        fnames = [f"{inspect.getmodule(x).__name__}.{x.__name__}"
                  for x in df['function']]
        df['canonical_function'] = fnames
        tmp = df.groupby('canonical_function')['status']
        print(tmp.value_counts())


@contextmanager
def numba_analysis():
    eval_hook_ctx, enable, disable, query = create_custom_hook(
        hook_name="numba_analysis_hook", frame_evaluator=numba_jit_analyser)

    handler = LogHandler(_log)
    with eval_hook_ctx():
        yield handler
