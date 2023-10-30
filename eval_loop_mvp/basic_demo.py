from pixie.eval_hook import create_custom_hook


def frame_evaluator(frame_info, co_extra):
    print("frame_evaluator cache on entry", co_extra)
    if 'cache' in co_extra:
        co_extra['cache'] += 1
    else:
        co_extra['cache'] = 1
    print("frame_evaluator cache on exit", co_extra)

    # Do not attempt to run `__exit__` functions
    ok = "__exit__" not in frame_info['f_code'].co_name
    print(f"Overridden interpreter running: '{frame_info['f_code'].co_name}', "
          f"accepted exec = {ok}")
    if ok:
        result = frame_info['f_func'](*frame_info['localsplus'])
        return True, result
    else:
        return False, None


def foo(x, a=11):
    # Test function
    return 10 * x, a


def demo():
    # Generate a frame evaluation hook, called `my_custom_hook`, that will use
    # the above frame_evaluator function to evaluate frames when the hook is
    # enabled.
    myhook, enable, disable, query = create_custom_hook(
        hook_name="my_custom_hook", frame_evaluator=frame_evaluator)

    # Run in the standard interpreter.
    query()
    print(foo(1))

    # Execute this region with the replacement frame evaluator.
    with myhook():
        query()
        print(foo(2), "\n\n")
        print(foo(3), "\n\n")
        print(foo(4), "\n\n")

    # Run in the standard interpreter.
    query()
    print(foo(5))


if __name__ == "__main__":
    demo()
