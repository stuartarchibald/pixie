"""Python C-API bindings"""
from llvmlite import ir
from pixie import llvm_types as lt


class RawPyAPI(object):

    @classmethod
    def PyImport_ImportModule(self, llvm_module):
        def PyImport_ImportModule_sig_type():
            signature = ir.FunctionType(lt._pyobject_head_p,
                                        # char *
                                        (lt._char_star,))

            name = "PyImport_ImportModule"
            return signature, name
        PyImport_ImportModule_fn = ir.Function(
            llvm_module, *PyImport_ImportModule_sig_type())
        PyImport_ImportModule_fn.linkage = 'external'
        return PyImport_ImportModule_fn

    @classmethod
    def PyObject_GetAttrString(self, llvm_module):
        def PyObject_GetAttrString_sig_type():
            signature = ir.FunctionType(lt._pyobject_head_p,
                                        # pyobj*, char *
                                        (lt._pyobject_head_p,
                                         lt._char_star,))
            name = "PyObject_GetAttrString"
            return signature, name

        PyObject_GetAttrString_fn = ir.Function(
            llvm_module, *PyObject_GetAttrString_sig_type())
        PyObject_GetAttrString_fn.linkage = 'external'
        return PyObject_GetAttrString_fn

    @classmethod
    def PyBytes_FromStringAndSize(self, llvm_module):
        def PyBytes_FromStringAndSize_sig_type():
            signature = ir.FunctionType(lt._pyobject_head_p,
                                        # char *, size_t
                                        (lt._char_star, lt._llvm_py_ssize_t))
            name = "PyBytes_FromStringAndSize"
            return signature, name
        PyBytes_FromStringAndSize_fn = ir.Function(
            llvm_module, *PyBytes_FromStringAndSize_sig_type())
        PyBytes_FromStringAndSize_fn.linkage = 'external'
        return PyBytes_FromStringAndSize_fn

    @classmethod
    def PyObject_CallFunctionObjArgs(self, llvm_module):
        def PyObject_CallFunctionObjArgs_sig_type():
            signature = ir.FunctionType(lt._pyobject_head_p,
                                        # pyobj*, ...
                                        (lt._pyobject_head_p,), var_arg=True)
            name = "PyObject_CallFunctionObjArgs"
            return signature, name

        PyObject_CallFunctionObjArgs_fn = ir.Function(
            llvm_module, *PyObject_CallFunctionObjArgs_sig_type())
        PyObject_CallFunctionObjArgs_fn.linkage = 'external'
        return PyObject_CallFunctionObjArgs_fn

    @classmethod
    def Py_BuildValue(self, llvm_module):
        def Py_BuildValue_sig_type():
            signature = ir.FunctionType(lt._pyobject_head_p,
                                        # const char *, ...
                                        (lt._char_star,),  var_arg=True)
            name = "Py_BuildValue"
            return signature, name

        Py_BuildValue_fn = ir.Function(llvm_module, *Py_BuildValue_sig_type())
        Py_BuildValue_fn.linkage = 'external'
        return Py_BuildValue_fn

    @classmethod
    def PyDict_SetItemString(self, llvm_module):
        def PyDict_SetItemString_sig_type():
            signature = ir.FunctionType(lt._int32,
                                        # (PyObject *p, const char *key,
                                        # PyObject *val)
                                        (lt._pyobject_head_p,
                                         lt._char_star,
                                         lt._pyobject_head_p), )
            name = "PyDict_SetItemString"
            return signature, name

        PyDict_SetItemString_fn = ir.Function(llvm_module,
                                              *PyDict_SetItemString_sig_type())
        PyDict_SetItemString_fn.linkage = 'external'
        return PyDict_SetItemString_fn

    @classmethod
    def PyDict_GetItemString(self, llvm_module):
        def PyDict_GetItemString_sig_type():
            signature = ir.FunctionType(lt._pyobject_head_p,
                                        # (PyObject *p, const char *key,)
                                        (lt._pyobject_head_p,
                                         lt._char_star,), )
            name = "PyDict_GetItemString"
            return signature, name

        PyDict_GetItemString_fn = ir.Function(llvm_module,
                                              *PyDict_GetItemString_sig_type())
        PyDict_GetItemString_fn.linkage = 'external'
        return PyDict_GetItemString_fn

    @classmethod
    def PyRun_String(self, llvm_module):
        def PyRun_String_sig_type():
            signature = ir.FunctionType(lt._pyobject_head_p,
                                        # const char *, int, pyobj *, pyobj*
                                        (lt._char_star,
                                         lt._int32,
                                         lt._pyobject_head_p,
                                         lt._pyobject_head_p), )
            name = "PyRun_String"
            return signature, name

        PyRun_String_fn = ir.Function(llvm_module, *PyRun_String_sig_type())
        PyRun_String_fn.linkage = 'external'
        return PyRun_String_fn

    @classmethod
    def PyUnicode_AsUTF8AndSize(self, llvm_module):
        def PyUnicode_AsUTF8AndSize_sig_type():
            signature = ir.FunctionType(lt._char_star,
                                        # pyobj *, py_ssize_t*
                                        (lt._pyobject_head_p,
                                         lt._llvm_py_ssize_t_star,),)
            name = "PyUnicode_AsUTF8AndSize"
            return signature, name
        args = PyUnicode_AsUTF8AndSize_sig_type()
        PyUnicode_AsUTF8AndSize_fn = ir.Function(llvm_module, *args)
        PyUnicode_AsUTF8AndSize_fn.linkage = 'external'
        return PyUnicode_AsUTF8AndSize_fn

    @classmethod
    def _PyTuple_Resize(self, llvm_module):
        def _PyTuple_Resize_sig_type():
            signature = ir.FunctionType(lt._int32,
                                        # void(pyobject**, py_ssize_t)
                                        (lt._pyobject_head_p.as_pointer(),
                                         lt._llvm_py_ssize_t,),)
            name = "_PyTuple_Resize"
            return signature, name
        args = _PyTuple_Resize_sig_type()
        _PyTuple_Resize_fn = ir.Function(llvm_module, *args)
        _PyTuple_Resize_fn.linkage = 'external'
        return _PyTuple_Resize_fn

    @classmethod
    def PyTuple_Size(self, llvm_module):
        def PyTuple_Size_sig_type():
            signature = ir.FunctionType(lt._llvm_py_ssize_t,
                                        # py_ssize_t(pyobject*)
                                        (lt._pyobject_head_p,),)
            name = "PyTuple_Size"
            return signature, name
        args = PyTuple_Size_sig_type()
        PyTuple_Size_fn = ir.Function(llvm_module, *args)
        PyTuple_Size_fn.linkage = 'external'
        return PyTuple_Size_fn

    @classmethod
    def Py_IncRef(self, llvm_module):
        def Py_IncRef_sig_type():
            signature = ir.FunctionType(ir.VoidType(),
                                        # pyobj *
                                        (lt._pyobject_head_p,),)
            name = "Py_IncRef"
            return signature, name
        args = Py_IncRef_sig_type()
        Py_IncRef_fn = ir.Function(llvm_module, *args)
        Py_IncRef_fn.linkage = 'external'
        return Py_IncRef_fn

    @classmethod
    def PyDict_New(self, llvm_module):
        def _PyDict_New_sig_type():
            signature = ir.FunctionType(lt._pyobject_head_p,
                                        # pyobject(void)
                                        (),)
            name = "PyDict_New"
            return signature, name
        args = _PyDict_New_sig_type()
        _PyDict_New_fn = ir.Function(llvm_module, *args)
        _PyDict_New_fn.linkage = 'external'
        return _PyDict_New_fn

    @classmethod
    def PyTuple_New(self, llvm_module):
        def _PyTuple_New_sig_type():
            signature = ir.FunctionType(lt._pyobject_head_p,
                                        # pyobject(py_ssize_t)
                                        (lt._llvm_py_ssize_t,),)
            name = "PyTuple_New"
            return signature, name
        args = _PyTuple_New_sig_type()
        _PyTuple_New_fn = ir.Function(llvm_module, *args)
        _PyTuple_New_fn.linkage = 'external'
        return _PyTuple_New_fn

    @classmethod
    def PyTuple_SetItem(self, llvm_module):
        def _PyTuple_SetItem_sig_type():
            signature = ir.FunctionType(lt._int32,
                                        # void(pyobject*, py_ssize_t, pyobject*)
                                        (lt._pyobject_head_p,
                                         lt._llvm_py_ssize_t,
                                         lt._pyobject_head_p),)
            name = "PyTuple_SetItem"
            return signature, name
        args = _PyTuple_SetItem_sig_type()
        _PyTuple_SetItem_fn = ir.Function(llvm_module, *args)
        _PyTuple_SetItem_fn.linkage = 'external'
        return _PyTuple_SetItem_fn

    @classmethod
    def PyTuple_GetItem(self, llvm_module):
        def _PyTuple_GetItem_sig_type():
            signature = ir.FunctionType(lt._pyobject_head_p,
                                        # void(pyobject*, py_ssize_t)
                                        (lt._pyobject_head_p,
                                         lt._llvm_py_ssize_t,),)
            name = "PyTuple_GetItem"
            return signature, name
        args = _PyTuple_GetItem_sig_type()
        _PyTuple_GetItem_fn = ir.Function(llvm_module, *args)
        _PyTuple_GetItem_fn.linkage = 'external'
        return _PyTuple_GetItem_fn

    @classmethod
    def _PyEval_RequestCodeExtraIndex(self, llvm_module):
        def _PyEval_RequestCodeExtraIndex_sig_type():
            # This function takes a freefunc, which is:
            # `void (*freefunc)(void*)`
            freefunc = ir.FunctionType(ir.VoidType(), (lt._void_star,))
            signature = ir.FunctionType(lt._int32,
                                        # pyobj *
                                        (freefunc.as_pointer(),),)
            name = "_PyEval_RequestCodeExtraIndex"
            return signature, name
        args = _PyEval_RequestCodeExtraIndex_sig_type()
        _PyEval_RequestCodeExtraIndex_fn = ir.Function(llvm_module, *args)
        _PyEval_RequestCodeExtraIndex_fn.linkage = 'external'
        return _PyEval_RequestCodeExtraIndex_fn

    @classmethod
    def _PyCode_GetExtra(self, llvm_module):
        def __PyCode_GetExtra_sig_type():
            signature = ir.FunctionType(lt._int32,
                                        # (pyobject *, py_ssize_t, void **)
                                        (lt._pyobject_head_p,
                                         lt._llvm_py_ssize_t,
                                         lt._void_star.as_pointer()),)
            name = "_PyCode_GetExtra"
            return signature, name
        args = __PyCode_GetExtra_sig_type()
        __PyCode_GetExtra_fn = ir.Function(llvm_module, *args)
        __PyCode_GetExtra_fn.linkage = 'external'
        return __PyCode_GetExtra_fn

    @classmethod
    def _PyCode_SetExtra(self, llvm_module):
        def __PyCode_SetExtra_sig_type():
            signature = ir.FunctionType(lt._int32,
                                        # (pyobject *, py_ssize_t, void *)
                                        (lt._pyobject_head_p,
                                         lt._llvm_py_ssize_t,
                                         lt._void_star),)
            name = "_PyCode_SetExtra"
            return signature, name
        args = __PyCode_SetExtra_sig_type()
        __PyCode_SetExtra_fn = ir.Function(llvm_module, *args)
        __PyCode_SetExtra_fn.linkage = 'external'
        return __PyCode_SetExtra_fn

    @classmethod
    def PyErr_SetString(self, llvm_module):
        def _PyErr_SetString_sig_type():
            signature = ir.FunctionType(ir.VoidType(),
                                        # (pyobject *, char *)
                                        (lt._pyobject_head_p,
                                         lt._char_star,),)
            name = "PyErr_SetString"
            return signature, name
        args = _PyErr_SetString_sig_type()
        _PyErr_SetString_fn = ir.Function(llvm_module, *args)
        _PyErr_SetString_fn.linkage = 'external'
        return _PyErr_SetString_fn
