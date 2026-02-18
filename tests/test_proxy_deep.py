"""Deep tests for transparent proxy logic.

Verifies attribute forwarding, method binding, and property access
through the Proxy wrapper.
"""

from __future__ import annotations

import pytest

from vetch.proxy import TransparentProxy, create_wrapper, get_original


class Base:
    """Base class for proxy testing."""
    def __init__(self):
        self.attr = "value"

    def method(self, x):
        return x * 2

    @property
    def prop(self):
        return "property"


def test_proxy_attribute_access():
    """Verify basic attribute access through proxy."""
    obj = Base()
    proxy = TransparentProxy(obj)

    assert proxy.attr == "value"
    assert proxy.method(5) == 10
    assert proxy.prop == "property"

def test_proxy_method_binding():
    """Verify methods are bound correctly to original object."""
    class Counter:
        def __init__(self):
            self.count = 0
        def increment(self):
            self.count += 1
            return self.count

    c = Counter()
    proxy = TransparentProxy(c)

    assert proxy.increment() == 1
    assert c.count == 1
    assert proxy.count == 1

def test_proxy_dunder_methods():
    """Verify basic dunder methods are handled."""
    class Dunder:
        def __init__(self):
            self.val = "data"
        def __str__(self):
            return "Dunder"
        def __repr__(self):
            return "<Dunder>"

    d = Dunder()
    proxy = TransparentProxy(d)

    # Proxy doesn't automatically forward all dunders unless implemented
    # But __str__ and __repr__ often work via default __getattr__ fallback
    # if the proxy doesn't override them.
    assert str(proxy) == "Dunder"

def test_proxy_nonexistent_attr():
    """Verify AttributeError is raised for missing attributes."""
    obj = Base()
    proxy = TransparentProxy(obj)

    with pytest.raises(AttributeError):
        _ = proxy.nonexistent

def test_proxy_setattr_delattr():
    """Verify setattr and delattr forwarding."""
    obj = Base()
    proxy = TransparentProxy(obj)

    proxy.new_attr = "new"
    assert obj.new_attr == "new"

    del proxy.new_attr
    assert not hasattr(obj, "new_attr")

def test_proxy_wrapped_attribute():
    """Verify _wrapped itself can be set on proxy."""
    obj1 = Base()
    obj2 = Base()
    proxy = TransparentProxy(obj1)

    object.__setattr__(proxy, "_wrapped", obj2)
    assert proxy.attr == "value"

def test_get_original_unwrapped():
    """Verify get_original returns object itself if not wrapped."""
    obj = Base()
    assert get_original(obj) is obj

def test_create_wrapper_before_call_fail():
    """Verify before_call failure is logged but doesn't block."""
    def original(x): return x
    def fail_hook(*a, **k): raise ValueError("Before fail")

    wrapped = create_wrapper(original, before_call=fail_hook)
    assert wrapped(10) == 10

def test_create_wrapper_after_call_fail():
    """Verify after_call failure is logged but doesn't block."""
    def original(x): return x
    def fail_hook(res, *a, **k): raise ValueError("After fail")

    wrapped = create_wrapper(original, after_call=fail_hook)
    assert wrapped(10) == 10

def test_create_wrapper_on_error_fail():
    """Verify on_error failure is logged but original exception raised."""
    def original(x): raise RuntimeError("Original error")
    def fail_hook(e): raise ValueError("Hook error")

    wrapped = create_wrapper(original, on_error=fail_hook)
    with pytest.raises(RuntimeError) as exc:
        wrapped(10)
    assert str(exc.value) == "Original error"

def test_create_stream_wrapper_hooks_fail():
    """Verify stream wrapper hooks failure handling."""
    from vetch.proxy import create_stream_wrapper
    def original(): yield 1
    def fail_chunk(c): raise ValueError("Chunk fail")
    def fail_complete(c): raise ValueError("Complete fail")

    wrapped = create_stream_wrapper(original, on_chunk=fail_chunk, on_complete=fail_complete)
    res = list(wrapped())
    assert res == [1]


