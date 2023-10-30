import atexit
import collections
import pickle

from contextlib import contextmanager
from ctypes import PYFUNCTYPE

from llvmlite import ir
import llvmlite.binding as llvm

from pixie import llvm_types as lt
from pixie import pyapi
from pixie.codegen_helpers import Context as CGCTX

int32 = lt._int32
i32_zero = int32(0)
void_star_star = lt._void_star.as_pointer()

_DEBUG = False

# ------------------------------------------------------------------------------
# BEGIN: Copied from Numba numba.core.cgutils.
# From:
# https://github.com/numba/numba/blob/fed996d3177b64fabf62a526fd329851819dc2da/numba/core/cgutils.py#L440-L443
# and
# https://github.com/numba/numba/blob/fed996d3177b64fabf62a526fd329851819dc2da/numba/core/cgutils.py#L472-L527
# ------------------------------------------------------------------------------


def terminate(builder, bbend):
    bb = builder.basic_block
    if bb.terminator is None:
        builder.branch(bbend)


def increment_index(builder, val):
    """
    Increment an index *val*.
    """
    one = val.type(1)
    # We pass the "nsw" flag in the hope that LLVM understands the index
    # never changes sign.  Unfortunately this doesn't always work
    # (e.g. ndindex()).
    return builder.add(val, one, flags=['nsw'])


Loop = collections.namedtuple('Loop', ('index', 'do_break'))


@contextmanager
def for_range(builder, count, start=None, intp=None):
    """
    Generate LLVM IR for a for-loop in [start, count).
    *start* is equal to 0 by default.

    Yields a Loop namedtuple with the following members:
    - `index` is the loop index's value
    - `do_break` is a no-argument callable to break out of the loop
    """
    if intp is None:
        intp = count.type
    if start is None:
        start = intp(0)
    stop = count

    bbcond = builder.append_basic_block("for.cond")
    bbbody = builder.append_basic_block("for.body")
    bbend = builder.append_basic_block("for.end")

    def do_break():
        builder.branch(bbend)

    bbstart = builder.basic_block
    builder.branch(bbcond)

    with builder.goto_block(bbcond):
        index = builder.phi(intp, name="loop.index")
        pred = builder.icmp_signed('<', index, stop)
        builder.cbranch(pred, bbbody, bbend)

    with builder.goto_block(bbbody):
        yield Loop(index, do_break)
        # Update bbbody as a new basic block may have been activated
        bbbody = builder.basic_block
        incr = increment_index(builder, index)
        terminate(builder, bbcond)

    index.add_incoming(start, bbstart)
    index.add_incoming(incr, bbbody)

    builder.position_at_end(bbend)
# ------------------------------------------------------------------------------
# END: Copied from Numba numba.core.cgutils.
# ------------------------------------------------------------------------------


def generate(name, eval_replacement_function):
    """
    Generate LLVM IR for functions that will provide mechanisms for replacing
    the evaluation loop function, namely:

    * enable_hook_{name}  - enable the eval replacement function.
    * disable_hook_{name} - disable (reset to default) the eval replacement
                            function.
    * query_hook_{name}   - print to STDOUT whether the default or a non-default
                            eval loop function is in use.

    Parameters:
    - name a reference name for the evaluation loop replacement that will be
      generated. It can be the same name as `eval_replacement_function`.
    - eval_replacement_function - a function serializable by pickle that will be
      called in place of the standard evaluation loop function.

    Returns:
    - a string representation of the LLVM module implementing the above.
    """

    mod = ir.Module()
    context = mod.context

    # --------------------------------------------------------------------------
    # Type declarations for `_ts` and `_if` are based on headers from
    # CPython.
    # https://github.com/python/cpython/blob/8e0c9213ac8d8ee8cb17c889aaabeafc40cb29f3/Include/internal/pycore_frame.h#L47-L67
    # and
    # https://github.com/python/cpython/blob/8e0c9213ac8d8ee8cb17c889aaabeafc40cb29f3/Include/cpython/pystate.h#L82-L203
    # --------------------------------------------------------------------------

    # threadstate refers to itself, so need to forward declare.
    _ts = context.get_identified_type('_py_threadstate')
    _ts_ptr = ir.PointerType(_ts)
    _py_threadstate_ptr = _ts_ptr

    # The interpreter state is a large and complex structure, the following only
    # ever uses it as a pointer arg to a function call so a void* will do.
    _py_interpreterstate_ptr = lt._void_star

    # interpreter frame refers to itself, so need to forward declare.
    _if = context.get_identified_type("_PyInterpreterFrame")
    _if_ptr = _if.as_pointer()
    _PyInterpreterFrame_ptr = _if_ptr

    # This is the frame evaluation function type i.e. type of the CPython
    # function _PyFrameEvalFunction.
    _PyFrameEvalFunction_ty = ir.FunctionType(lt._pyobject_head_p,
                                              (_py_threadstate_ptr,  # tstate
                                               _PyInterpreterFrame_ptr,  # frame
                                               int32)).as_pointer()

    # This is PyThreadState
    # See:
    # https://github.com/python/cpython/blob/8e0c9213ac8d8ee8cb17c889aaabeafc40cb29f3/Include/cpython/pystate.h#L82-L203
    _ts.set_body(_ts_ptr,
                 _ts_ptr,
                 _py_interpreterstate_ptr,
                 int32,
                 lt._void_star)

    _Py_CODEUNIT_ptr = ir.IntType(16).as_pointer()  # it's a uint16_t*.

    # This is _PyInterpreterFrame
    # See:
    # https://github.com/python/cpython/blob/8e0c9213ac8d8ee8cb17c889aaabeafc40cb29f3/Include/internal/pycore_frame.h#L47-L67
    _if.set_body(lt._pyobject_head_p,  # f_func, strong ref
                 lt._pyobject_head_p,  # f_globals, borrowed ref
                 lt._pyobject_head_p,  # f_builtins, borrowed ref
                 lt._pyobject_head_p,  # f_locals, strong ref, may be NULL
                 lt._pyobject_head_p,  # f_code, strong ref
                 lt._pyobject_head_p,  # frame_obj, strong ref, may be NULL
                 _if_ptr,  # previous
                 _Py_CODEUNIT_ptr,  # prev_instr
                 int32,  # stacktop
                 ir.IntType(8),  # is_entry
                 ir.IntType(8),  # owner
                 ir.ArrayType(lt._pyobject_head_p, 1),  # locals plus
                 )

    def PyThreadState_Get(llvm_module):
        def PyThreadState_Get_sig_type():
            signature = ir.FunctionType(_ts_ptr,
                                        # no args
                                        (),)
            name = "PyThreadState_Get"
            return signature, name

        PyThreadState_Get_fn = ir.Function(llvm_module,
                                           *PyThreadState_Get_sig_type())
        PyThreadState_Get_fn.linkage = 'external'
        return PyThreadState_Get_fn

    def __PyEval_EvalFrameDefault(llvm_module):
        def __PyEval_EvalFrameDefault_sig_type():
            signature = _PyFrameEvalFunction_ty.pointee
            name = "_PyEval_EvalFrameDefault"
            return signature, name
        args = __PyEval_EvalFrameDefault_sig_type()
        __PyEval_EvalFrameDefault_fn = ir.Function(llvm_module, *args)
        __PyEval_EvalFrameDefault_fn.linkage = 'external'
        return __PyEval_EvalFrameDefault_fn

    def __PyInterpreterState_GetEvalFrameFunc(llvm_module):
        def __PyInterpreterState_GetEvalFrameFunc_sig_type():
            signature = ir.FunctionType(_PyFrameEvalFunction_ty,
                                        # void(PyInterpreterState *)
                                        (_py_interpreterstate_ptr,),)
            name = "_PyInterpreterState_GetEvalFrameFunc"
            return signature, name
        args = __PyInterpreterState_GetEvalFrameFunc_sig_type()
        __PyInterpreterState_GetEvalFrameFunc_fn = ir.Function(llvm_module,
                                                               *args)
        __PyInterpreterState_GetEvalFrameFunc_fn.linkage = 'external'
        return __PyInterpreterState_GetEvalFrameFunc_fn

    def __PyInterpreterState_SetEvalFrameFunc(llvm_module):
        def __PyInterpreterState_SetEvalFrameFunc_sig_type():
            signature = ir.FunctionType(ir.VoidType(),
                                        # void(PyInterpreterState *,
                                        # the frame eval func type is defined as
                                        # a pointer
                                        # _PyFrameEvalFunction)
                                        (_py_interpreterstate_ptr,
                                         _PyFrameEvalFunction_ty),)
            name = "_PyInterpreterState_SetEvalFrameFunc"
            return signature, name
        args = __PyInterpreterState_SetEvalFrameFunc_sig_type()
        __PyInterpreterState_SetEvalFrameFunc_fn = ir.Function(llvm_module,
                                                               *args)
        __PyInterpreterState_SetEvalFrameFunc_fn.linkage = 'external'
        return __PyInterpreterState_SetEvalFrameFunc_fn

    cgctx = CGCTX()
    py_none_glbl = cgctx.add_global_variable(mod, ir.IntType(8),
                                             "_Py_NoneStruct")
    py_true_glbl = cgctx.add_global_variable(mod, ir.IntType(8),
                                             "_Py_TrueStruct")
    PyExc_ValueError = cgctx.add_global_variable(mod, lt._pyobject_head_p,
                                                 "PyExc_ValueError")

    PyObject_NULL = cgctx.get_null_value(lt._pyobject_head_p)

    _PyEval_EvalFrameDefault = __PyEval_EvalFrameDefault(mod)
    _PyInterpreterState_SetEvalFrameFunc = \
        __PyInterpreterState_SetEvalFrameFunc(mod)

    py_api = pyapi.RawPyAPI()
    PyImport_ImportModule = py_api.PyImport_ImportModule(mod)
    PyObject_GetAttrString = py_api.PyObject_GetAttrString(mod)
    PyBytes_FromStringAndSize = py_api.PyBytes_FromStringAndSize(mod)
    PyObject_CallFunctionObjArgs = py_api.PyObject_CallFunctionObjArgs(mod)
    PyDict_SetItemString = py_api.PyDict_SetItemString(mod)
    Py_IncRef = py_api.Py_IncRef(mod)
    PyDict_New = py_api.PyDict_New(mod)
    PyTuple_New = py_api.PyTuple_New(mod)
    PyTuple_GetItem = py_api.PyTuple_GetItem(mod)
    PyTuple_SetItem = py_api.PyTuple_SetItem(mod)
    PyTuple_Size = py_api.PyTuple_Size(mod)
    _PyTuple_Resize = py_api._PyTuple_Resize(mod)
    _PyEval_RequestCodeExtraIndex = py_api._PyEval_RequestCodeExtraIndex(mod)
    _PyCode_GetExtra = py_api._PyCode_GetExtra(mod)
    _PyCode_SetExtra = py_api._PyCode_SetExtra(mod)
    PyErr_SetString = py_api.PyErr_SetString(mod)

    # --------------------------------------------------------------------------
    # Start custom eval frame function
    # --------------------------------------------------------------------------

    # Custom eval frame function, the type of the function is the function
    # pointed to by _PyFrameEvalFunction_ty (which is defined as a function
    # pointer).
    func = ir.Function(mod, _PyFrameEvalFunction_ty.pointee, name)
    blk = func.append_basic_block()
    builder = ir.IRBuilder(blk)
    if _DEBUG:
        cgctx.printf(builder, "\nRunning custom frame evaluator!\n")

    # unpack the args into locals.
    def args2locals(builder, args):
        ret = []
        for i in range(len(func.args)):
            slot = builder.alloca(func.args[i].type)
            builder.store(func.args[i], slot)
            ret.append(slot)
        return ret

    ts_ptr, frame_ptr, throwflag = args2locals(builder, func.args)

    # get the code object ptr from the frame
    ts = builder.load(ts_ptr)
    frame = builder.load(frame_ptr)
    co = builder.gep(frame, [int32(x) for x in (0, 4)])

    state = builder.alloca(lt._pyobject_head_p)
    # store null into the state, this signals not set
    builder.store(cgctx.get_null_value(state.type.pointee), state)

    # create and assign in codeextraindex slots, this should be in
    # per-interpreter storage but it currently isn't.
    py_n_code_indexes_arrty = ir.ArrayType(int32, 1)
    extra_code_indexes = cgctx.add_global_variable(mod, py_n_code_indexes_arrty,
                                                   "_extra_code_indexes")
    extra_code_indexes.linkage = 'dso_local'
    extra_code_indexes.initializer = ir.Constant(py_n_code_indexes_arrty, [-1,])

    extra_code_index1_raw = builder.load(builder.gep(extra_code_indexes,
                                                     (i32_zero, i32_zero)))
    extra_code_index1 = builder.sext(extra_code_index1_raw, lt._llvm_py_ssize_t)
    state_void_ptr_ptr = builder.bitcast(state, void_star_star)
    stat = builder.call(_PyCode_GetExtra, (builder.load(co), extra_code_index1,
                                           state_void_ptr_ptr))

    # Set up co_extra slot with a dict for arbitrary cache use.
    # If call stat is zero and the state is NULL then there's no cache set yet
    # as a result one needs to be set up!
    stat_pred = builder.icmp_signed('==', stat, i32_zero)
    state_pred = builder.icmp_unsigned('==', builder.load(state),
                                       cgctx.get_null_value(state.type))
    all_pred = builder.and_(stat_pred, state_pred)
    with builder.if_else(all_pred) as (then, otherwise):
        with then:
            if _DEBUG:
                cgctx.printf(builder, "co_extra cache miss\n")

            co_extra_cache = builder.call(PyDict_New, ())
            with builder.if_then(cgctx.is_null(builder, co_extra_cache)):
                cgctx.printf(builder, "PyDict_New failed\n")
                builder.ret(PyObject_NULL)
            builder.store(co_extra_cache, state)
            state_void_ptr = builder.bitcast(builder.load(state), lt._void_star)
            tmp = builder.call(_PyCode_SetExtra, (builder.load(co),
                                                  extra_code_index1,
                                                  state_void_ptr))
            with builder.if_then(builder.icmp_signed("!=", tmp, i32_zero)):
                cgctx.printf(builder, "_PyCode_SetExtra failed, %d\n",
                             extra_code_index1)
                builder.ret(PyObject_NULL)
            if _DEBUG:
                cgctx.printf(builder,
                             ("after _PyCode_SetExtra in then: stat = %d,"
                              "state = %d\n"), stat, builder.load(state))
                tmp = builder.call(_PyCode_GetExtra,
                                   (builder.load(co), extra_code_index1,
                                    builder.bitcast(state,
                                                    void_star_star)))
                with builder.if_then(builder.icmp_signed("!=", tmp, i32_zero)):
                    cgctx.printf(builder, "_PyCode_GetExtra failed\n")
                    builder.ret(PyObject_NULL)
                cgctx.printf(builder,
                             ("after _PyCode_GetExtra in then: stat = %d, "
                              "state = %d\n"), stat, builder.load(state))
        with otherwise:
            if _DEBUG:
                cgctx.printf(builder, "co_extra \n")

    # The co_extra cache is now set up, now work on the frame.
    interp_field = builder.gep(ts, [int32(x) for x in (0, 2)])
    loaded_interp_field = builder.load(interp_field)

    # store a reference to the current frame interpreter
    _PyInterpreterState_GetEvalFrameFunc = \
        __PyInterpreterState_GetEvalFrameFunc(mod)
    original_evaluator = builder.alloca(_PyFrameEvalFunction_ty)
    builder.store(builder.call(_PyInterpreterState_GetEvalFrameFunc,
                               (loaded_interp_field,)), original_evaluator)
    # use the stock interpreter to evaluate the python payload
    builder.call(_PyInterpreterState_SetEvalFrameFunc,
                 (loaded_interp_field, _PyEval_EvalFrameDefault))

    # this unpickles and loads the user defined eval replacement function
    pickled_payload = pickle.dumps(eval_replacement_function)
    payload = cgctx.insert_const_bytes(mod, pickled_payload)
    sz = len(pickled_payload)
    len_payload = cgctx.global_constant(mod, "_len_payload",
                                        lt._llvm_py_ssize_t(sz))

    py_mod_name = cgctx.insert_const_string(mod, "pickle")
    py_mod = builder.call(PyImport_ImportModule, (py_mod_name,))
    with builder.if_then(cgctx.is_null(builder, py_mod)):
        cgctx.printf(builder, "PyImport_ImportModule failed\n")
        builder.ret(PyObject_NULL)

    bounce_func_name = cgctx.insert_const_string(mod, "loads")
    bounce_func = builder.call(PyObject_GetAttrString, (py_mod,
                                                        bounce_func_name))
    with builder.if_then(cgctx.is_null(builder, bounce_func)):
        cgctx.printf(builder, "PyObject_GetAttrString failed\n")
        builder.ret(PyObject_NULL)

    payload_pybytes = builder.call(PyBytes_FromStringAndSize,
                                   (payload, builder.load(len_payload)))
    with builder.if_then(cgctx.is_null(builder, payload_pybytes)):
        cgctx.printf(builder, "PyObject_CallFunctionObjArgs failed\n")
        builder.ret(PyObject_NULL)

    deserialized_payload = builder.call(PyObject_CallFunctionObjArgs,
                                        (bounce_func, payload_pybytes,
                                         PyObject_NULL))
    with builder.if_then(cgctx.is_null(builder, deserialized_payload)):
        cgctx.printf(builder, "PyObject_CallFunctionObjArgs failed\n")
        builder.ret(PyObject_NULL)

    # Create dictionary of "stuff" that's a bit like the frame to pass into the
    # user defined eval replacement function
    frame_dict = builder.call(PyDict_New, ())
    with builder.if_then(cgctx.is_null(builder, frame_dict)):
        cgctx.printf(builder, "PyDict_New failed\n")
        builder.ret(PyObject_NULL)

    frame_keys = {'f_func': 0,
                  'f_globals': 1,
                  'f_builtins': 2,
                  'f_locals': 3,
                  'f_code': 4,
                  'frame_obj': 5
                  }
    py_none = builder.bitcast(py_none_glbl, lt._pyobject_head_p)
    for k, idx in frame_keys.items():
        frame_func_addr = builder.gep(frame, [int32(x) for x in (0, idx)])
        frame_func = builder.load(frame_func_addr)
        # see if the thing is NULL, it might be
        func_str = cgctx.insert_const_string(mod, k)
        if _DEBUG:
            cgctx.printf(builder, "func_str %s\n", func_str)
            cgctx.printf(builder, "frame_func_addr IS NULL %d\n",
                         cgctx.is_null(builder, frame_func_addr))
            cgctx.printf(builder, "IS NULL %d\n",
                         cgctx.is_null(builder, frame_func))
        with builder.if_else(cgctx.is_null(builder, frame_func)) as \
                (l_then, l_otherwise):
            with l_then:
                if _DEBUG:
                    cgctx.printf(builder, "%s is null\n", func_str)
                builder.call(Py_IncRef, [py_none,])
                stat = builder.call(PyDict_SetItemString, (frame_dict, func_str,
                                                           py_none))
                with builder.if_then(builder.icmp_signed("!=", stat, i32_zero)):
                    if _DEBUG:
                        cgctx.printf(builder, "PyDict_SetItemString failed\n")
                    builder.ret(PyObject_NULL)
            with l_otherwise:
                if _DEBUG:
                    cgctx.printf(builder, "%s is not null\n", func_str)
                builder.call(Py_IncRef,(frame_func,))
                stat = builder.call(PyDict_SetItemString, (frame_dict, func_str,
                                                           frame_func))
                with builder.if_then(builder.icmp_signed("!=", stat, i32_zero)):
                    cgctx.printf(builder, "PyDict_SetItemString failed\n")
                    builder.ret(PyObject_NULL)
    # Need to get the localsplus into the dict, it has an item count of up to
    # `stacktop`s value. Allocate a tuple `stacktop` in size, start filling
    # in the tuple with the values until a NULL value is hit, then truncate
    # the tuple to that count in size.
    stacktop_key = 8 # it's item 8 in the frame struct
    frame_stacktop = builder.gep(frame, [int32(x) for x in (0, stacktop_key)])
    stacktop_size = builder.load(frame_stacktop)
    # get a tuple that can hold up to stacktop size things (it'll get
    # truncated to the right size later).
    tup = builder.call(PyTuple_New,
                       (builder.sext(stacktop_size, lt._llvm_py_ssize_t),))
    with builder.if_then(cgctx.is_null(builder, tup)):
        cgctx.printf(builder, "PyTuple_New failed\n")
        builder.ret(PyObject_NULL)
    localsplus_key = 11
    frame_localsplus = builder.gep(frame,
                                   [int32(x) for x in (0, localsplus_key)])
    if _DEBUG:
        cgctx.printf(builder, "stacktop_size is %d\n", stacktop_size)
    with for_range(builder, count=stacktop_size) as (index, escape):
        local = builder.load(builder.gep(frame_localsplus, [i32_zero, index,]))
        pred = cgctx.is_null(builder, local)
        with builder.if_then(pred):
            if _DEBUG:
                cgctx.printf(builder, "early exit for localsplus\n")
            escape()
        builder.call(Py_IncRef, [local,])
        stat = builder.call(PyTuple_SetItem, (tup,
                                              builder.sext(index,
                                                           lt._llvm_py_ssize_t),
                                              local))
        with builder.if_then(builder.icmp_signed("!=", stat, i32_zero)):
            cgctx.printf(builder, "PyTuple_SetItem failed\n")
            builder.ret(PyObject_NULL)
    tup_ptr = builder.alloca(lt._pyobject_head_p)
    builder.store(tup, tup_ptr)
    extended_idx = builder.sext(index, lt._llvm_py_ssize_t)
    if _DEBUG:
        cgctx.printf(builder, "localsplus size %d\n", extended_idx)
    # This call potentially clobbers `tup` as a write through `tup_ptr`,
    # future use of `tup` needs to be through a load of `tup_ptr`.
    stat = builder.call(_PyTuple_Resize, (tup_ptr, extended_idx))
    with builder.if_then(builder.icmp_signed("!=", stat, i32_zero)):
        cgctx.printf(builder, "_PyTuple_Resize failed\n")
        builder.ret(PyObject_NULL)
    func_str = cgctx.insert_const_string(mod, "localsplus")
    stat = builder.call(PyDict_SetItemString, (frame_dict, func_str,
                                               builder.load(tup_ptr)))
    with builder.if_then(builder.icmp_signed("!=", stat, i32_zero)):
        cgctx.printf(builder, "PyDict_SetItemString failed\n")
        builder.ret(PyObject_NULL)

    # Finally... make the call to the user defined eval loop replacement
    # function, it takes two args, 1. the "frame" dict, 2. the "cache" dict
    # which is in the code object extran index for the current frame.
    result = builder.call(PyObject_CallFunctionObjArgs,
                          (deserialized_payload, frame_dict,
                           builder.load(state), PyObject_NULL))
    with builder.if_then(cgctx.is_null(builder, result)):
        cgctx.printf(builder, "PyObject_CallFunctionObjArgs failed\n")
        builder.ret(PyObject_NULL)

    if _DEBUG:
        pred = builder.icmp_unsigned("==", builder.load(original_evaluator),
                                     _PyEval_EvalFrameDefault)
        with builder.if_else(pred) as (then, otherwise):
            with then:
                cgctx.printf(builder, ("resetting frame eval to ORIGINAL which "
                                       "is DEFAULT\n"))
            with otherwise:
                cgctx.printf(builder,
                             ("resetting to frame eval ORIGINAL which is "
                              "non-DEFAULT\n"))

    # Set the interpreter back to the original one (this function)
    builder.call(_PyInterpreterState_SetEvalFrameFunc,
                 (loaded_interp_field, builder.load(original_evaluator)))

    # Now check what the user defined eval loop replacement function returned.
    # If result[0] is True, the user eval frame function handled evaluation and
    # the result[1] is the "answer", if result[0] is False the user eval frame
    # function failed and so the standard __PyEval_EvalFrameDefault function
    # needs calling.

    # first check that a 2-tuple has been returned
    result_len = builder.call(PyTuple_Size, (result,))
    len_is_2 = builder.icmp_signed("==", result_len, lt._llvm_py_ssize_t(2))
    with builder.if_then(builder.not_(len_is_2)):
        s = f"Expected eval loop hook '{name}' to return a 2-tuple."
        msg = cgctx.insert_const_string(mod, s)
        builder.call(PyErr_SetString, (builder.load(PyExc_ValueError), msg))
        builder.ret(PyObject_NULL)

    # Fetch item 0 in the result tuple
    eval_success = builder.call(PyTuple_GetItem,
                                (result, ir.Constant(lt._llvm_py_ssize_t, 0)))
    with builder.if_then(cgctx.is_null(builder, eval_success)):
        cgctx.printf(builder, "PyTuple_GetItem failed\n")
        builder.ret(PyObject_NULL)
    py_true = builder.bitcast(py_true_glbl, lt._pyobject_head_p)
    eval_success_pred = builder.icmp_signed("==", eval_success, py_true)
    py_return = builder.alloca(lt._pyobject_head_p)
    # if return[0] is True, then... otherwise...
    with builder.if_else(eval_success_pred) as (then, otherwise):
        with then:
            # Success: set the return value to return[1]
            if _DEBUG:
                cgctx.printf(builder, "evaluated OK with NON-DEFAULT\n")
            tmp = builder.call(PyTuple_GetItem,
                               (result, ir.Constant(lt._llvm_py_ssize_t, 1)))
            with builder.if_then(cgctx.is_null(builder, tmp)):
                cgctx.printf(builder, "PyTuple_GetItem failed\n")
                builder.ret(PyObject_NULL)
            builder.store(tmp, py_return)
        with otherwise:
            # Failure: Need to evaluate the frame with the standard eval frame
            # function.
            if _DEBUG:
                cgctx.printf(builder,
                             "evaluating with DEFAULT as NON-DEFAULT failed\n")
            tmp = builder.call(_PyEval_EvalFrameDefault,
                               (builder.load(ts_ptr), builder.load(frame_ptr),
                                builder.load(throwflag)))
            with builder.if_then(cgctx.is_null(builder, tmp)):
                cgctx.printf(builder,
                             "call to _PyEval_EvalFrameDefault failed\n")
                builder.ret(PyObject_NULL)
            builder.store(tmp, py_return)

    builder.ret(builder.load(py_return))
    # --------------------------------------------------------------------------
    # End custom eval frame function
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Start enable hook function
    # --------------------------------------------------------------------------

    # This function sets the eval frame slot on the interpreter state to the
    # custom eval frame function.
    enable_fnty = ir.FunctionType(lt._pyobject_head_p, (lt._pyobject_head_p,))
    enable_func = ir.Function(mod, enable_fnty, f"enable_hook_{name}")
    blk = enable_func.append_basic_block()
    builder = ir.IRBuilder(blk)

    # check if the hook is already enabled and warn
    hook_enabled = cgctx.add_global_variable(mod, ir.IntType(1),
                                             f"_hook_{name}_enabled")
    hook_enabled.initializer = ir.Constant(ir.IntType(1), 0)
    with builder.if_else(builder.load(hook_enabled)) as (then, otherwise):
        with then:
            cgctx.printf(builder, (f"WARNING: Eval loop hook '{name}' is "
                                   "already enabled.\n"))
        with otherwise:
            builder.store(ir.IntType(1)(1), hook_enabled)

    # make the requests
    null_freefunc = _PyEval_RequestCodeExtraIndex.args[0].type(None)
    index = builder.call(_PyEval_RequestCodeExtraIndex, [null_freefunc,])
    builder.store(index, builder.gep(extra_code_indexes, [i32_zero, i32_zero]))

    PyThreadState_Get_fn = PyThreadState_Get(mod)
    ts = builder.call(PyThreadState_Get_fn, ())
    ts_slot = builder.alloca(_py_threadstate_ptr)
    builder.store(ts, ts_slot)
    ts_from_slot = builder.load(ts_slot)

    interp_field = builder.gep(ts_from_slot, [int32(x) for x in (0, 2)])
    loaded_interp_field = builder.load(interp_field)
    # Avoid doing this the "C" way, the interpreter state structure is
    # complicated, use the unstable CPython API instead.
    builder.call(_PyInterpreterState_SetEvalFrameFunc, (loaded_interp_field,
                                                        func))
    if _DEBUG:
        cgctx.printf(builder, "In enable hook!\n")
    py_none = builder.bitcast(py_none_glbl, lt._pyobject_head_p)
    builder.call(Py_IncRef, [py_none,])
    builder.ret(py_none)
    # --------------------------------------------------------------------------
    # End enable hook function
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Start disable hook function
    # --------------------------------------------------------------------------

    # This function sets the eval frame slot on the interpreter state to the
    # standard eval frame function (it effectively disables the custom hook).
    enable_fnty = ir.FunctionType(lt._pyobject_head_p, (lt._pyobject_head_p,))
    enable_func = ir.Function(mod, enable_fnty, f"disable_hook_{name}")
    blk = enable_func.append_basic_block()
    builder = ir.IRBuilder(blk)
    if _DEBUG:
        cgctx.printf(builder, "Entering disable hook!\n")

    ts = builder.call(PyThreadState_Get_fn, ())
    ts_slot = builder.alloca(_py_threadstate_ptr)
    builder.store(ts, ts_slot)
    ts_from_slot = builder.load(ts_slot)

    interp_field = builder.gep(ts_from_slot, [int32(x) for x in (0, 2)])
    loaded_interp_field = builder.load(interp_field)
    # Avoid doing this the "C" way, the interpreter state structure is
    # complicated, use the unstable CPython API instead.
    builder.call(_PyInterpreterState_SetEvalFrameFunc,
                 (loaded_interp_field, _PyEval_EvalFrameDefault))

    py_none = builder.bitcast(py_none_glbl, lt._pyobject_head_p)
    builder.call(Py_IncRef, [py_none,])
    if _DEBUG:
        cgctx.printf(builder, "Exiting disable hook!\n")
    builder.ret(py_none)
    # --------------------------------------------------------------------------
    # End disable hook function
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Start query hook function
    # --------------------------------------------------------------------------

    # This function prints the current eval frame hook as "DEFAULT" for the
    # default one or "NON-DEFAULT" for a custom one.
    query_fnty = ir.FunctionType(lt._pyobject_head_p, (lt._pyobject_head_p,))
    query_func = ir.Function(mod, query_fnty, f"query_hook_{name}")
    blk = query_func.append_basic_block()
    builder = ir.IRBuilder(blk)
    ts = builder.call(PyThreadState_Get_fn, ())
    ts_slot = builder.alloca(_py_threadstate_ptr)
    builder.store(ts, ts_slot)
    ts_from_slot = builder.load(ts_slot)

    interp_field = builder.gep(ts_from_slot, [int32(x) for x in (0, 2)])
    loaded_interp_field = builder.load(interp_field)
    current_evaluator = builder.call(_PyInterpreterState_GetEvalFrameFunc,
                                     (loaded_interp_field,))

    pred = builder.icmp_unsigned("==", current_evaluator,
                                 _PyEval_EvalFrameDefault)
    with builder.if_else(pred) as (then, otherwise):
        with then:
            cgctx.printf(builder, "Interpreter: DEFAULT, %x\n",
                         loaded_interp_field)
        with otherwise:
            cgctx.printf(builder, "Interpreter: NON-DEFAULT %x\n",
                         loaded_interp_field)
    py_none = builder.bitcast(py_none_glbl, lt._pyobject_head_p)
    builder.call(Py_IncRef, [py_none,])
    builder.ret(py_none)
    # --------------------------------------------------------------------------
    # End query hook function
    # --------------------------------------------------------------------------

    # Return the LLVM module as a string.
    return str(mod)

# ------------------------------------------------------------------------------
# BEGIN: Adapted from llvmlite
#
# https://github.com/numba/llvmlite/blob/f22420ad768a31dd15ae45d551bb4e73907a27fa/docs/source/user-guide/examples/ll_fpadd.py#L9-L11
# and
# https://github.com/numba/llvmlite/blob/f22420ad768a31dd15ae45d551bb4e73907a27fa/docs/source/user-guide/examples/ll_fpadd.py#L26-L53
# ------------------------------------------------------------------------------


llvm.initialize()


llvm.initialize_native_target()


llvm.initialize_native_asmprinter()


def _create_execution_engine():
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
    return engine


def _compile_ir(engine, llvm_ir):
    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()
    return mod


_engine = _create_execution_engine()

# ------------------------------------------------------------------------------
# END: Adapted from llvmlite
# ------------------------------------------------------------------------------


def create_custom_hook(*, hook_name, frame_evaluator):
    """
    Create a set of functions to help with custom frame evaluation.

    This generates a set of helper functions to allow a user to replace the
    current Python interpreter with a supplied Python function. This function
    could be used to simply analyse the executing frames right through to
    implementing a JIT compiler for the Python interpreter. If the supplied
    function declares "success" then it is assumed that the function executed
    the frame correctly and the result is directly consumed. If the supplied
    function declares "failure" then it is assumed that the function could not
    execute the frame and the standard interpreter will be called instead so as
    to obtain a result and preserve execution semantics.

    Parameters:
        - hook_name [str] the name of the hook
        - frame_evaluator [function] the custom frame evaluator.

    The `frame_evaluator` function must comply with the following:
    1. It must have the signature `frame_evaluator(frame_info, cache)`, where
       frame_info is a dictionary containing information about the current frame
       that is wired in directly from the interpreter state, and cache is a
       dictionary for use by the user for e.g. storing a JIT-compiled version of
       this frame.
    2. It must be serializable with `pickle`.
    3. It must return a 2-tuple (success [bool], result [value or None]). If the
      evaluation is successful then success must be True and the result the
      result of the evaluation. If the evaluation is a failure then success
      must be False and the result is None.

    Returns:
        - a tuple comprising:
            - a context manager that will replace the current frame evaluator
              with the frame_evaluator function for the duration of the context
              managed region.
            - an `enable` function, to manually enable the use of the
              `frame_evaluator` in place of the current frame evaluator.
            - a `disable` function, to manually set the interpreter to the
              default CPython frame evaluator.
            - a `query` function, which prints whether the current frame
              evaluator is the default CPython interpreter or something else.
    """
    llvm_ir = generate(hook_name, frame_evaluator)
    _compile_ir(_engine, llvm_ir)
    func_ptr_enable = _engine.get_function_address(f"enable_hook_{hook_name}")
    cfunc_register_hook = PYFUNCTYPE(None, )(func_ptr_enable)
    func_ptr_disable = _engine.get_function_address(f"disable_hook_{hook_name}")
    cfunc_deregister_hook = PYFUNCTYPE(None, )(func_ptr_disable)
    func_ptr_query = _engine.get_function_address(f"query_hook_{hook_name}")
    cfunc_query_hook = PYFUNCTYPE(None, )(func_ptr_query)

    # make sure hooks are always deregistered at exit, if this is forgotten or
    # omitted a confusing situation can occur where the hooks are being garbage
    # collected/unloaded whilst also being used to run the shutdown.
    def atexit_disable():
        if _DEBUG:
            print("At exit disabling hook")
        cfunc_deregister_hook()
    atexit.register(cfunc_deregister_hook)

    class hook():

        def __enter__(self):
            cfunc_register_hook()

        def __exit__(self, *args, **kwargs):
            cfunc_deregister_hook()

    return hook, cfunc_register_hook, cfunc_deregister_hook, cfunc_query_hook
