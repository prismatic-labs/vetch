"""Tests for proxy module."""

from typing import Any

from vetch.proxy import (
    TransparentProxy,
    create_stream_wrapper,
    create_wrapper,
    get_original,
    is_vetch_patched,
)


class TestTransparentProxy:
    """Tests for TransparentProxy class."""

    def test_forwards_attribute_access(self) -> None:
        """Proxy forwards attribute access to wrapped object."""

        class Target:
            value = 42
            name = "test"

        target = Target()
        proxy = TransparentProxy(target)
        assert proxy.value == 42
        assert proxy.name == "test"

    def test_forwards_method_calls(self) -> None:
        """Proxy forwards method calls."""

        class Target:
            def greet(self, name: str) -> str:
                return f"Hello, {name}!"

        target = Target()
        proxy = TransparentProxy(target)
        assert proxy.greet("World") == "Hello, World!"

    def test_callable_proxy(self) -> None:
        """Proxy can wrap callable objects."""

        def target(x: int) -> int:
            return x * 2

        proxy = TransparentProxy(target)
        assert proxy(5) == 10


class TestCreateWrapper:
    """Tests for create_wrapper function."""

    def test_preserves_function_behavior(self) -> None:
        """Wrapped function behaves like original."""

        def add(a: int, b: int) -> int:
            return a + b

        wrapped = create_wrapper(add)
        assert wrapped(2, 3) == 5

    def test_marks_as_patched(self) -> None:
        """Wrapped function is marked as patched."""

        def func() -> None:
            pass

        wrapped = create_wrapper(func)
        assert is_vetch_patched(wrapped) is True
        assert is_vetch_patched(func) is False

    def test_stores_original(self) -> None:
        """Wrapped function stores reference to original."""

        def func() -> None:
            pass

        wrapped = create_wrapper(func)
        assert get_original(wrapped) is func

    def test_before_call_hook(self) -> None:
        """before_call hook is invoked."""
        calls: list[str] = []

        def func() -> str:
            return "result"

        def before(*args: Any, **kwargs: Any) -> None:
            calls.append("before")

        wrapped = create_wrapper(func, before_call=before)
        result = wrapped()
        assert result == "result"
        assert calls == ["before"]

    def test_after_call_hook(self) -> None:
        """after_call hook is invoked with result."""
        results: list[str] = []

        def func() -> str:
            return "result"

        def after(result: str, *args: Any, **kwargs: Any) -> None:
            results.append(result)

        wrapped = create_wrapper(func, after_call=after)
        wrapped()
        assert results == ["result"]

    def test_on_error_hook(self) -> None:
        """on_error hook is invoked on exception."""
        errors: list[str] = []

        def func() -> None:
            raise ValueError("test error")

        def on_error(e: BaseException) -> None:
            errors.append(type(e).__name__)

        wrapped = create_wrapper(func, on_error=on_error)

        try:
            wrapped()
        except ValueError:
            pass

        assert errors == ["ValueError"]

    def test_exception_propagates(self) -> None:
        """Exceptions are not suppressed."""

        def func() -> None:
            raise RuntimeError("test")

        wrapped = create_wrapper(func)

        try:
            wrapped()
            raise AssertionError("Should have raised")
        except RuntimeError as e:
            assert str(e) == "test"

    def test_hook_failure_does_not_block(self) -> None:
        """Hook failures don't block the original function."""
        results: list[str] = []

        def func() -> str:
            results.append("func")
            return "ok"

        def bad_before(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("hook failed")

        wrapped = create_wrapper(func, before_call=bad_before)
        result = wrapped()

        assert result == "ok"
        assert results == ["func"]


class TestCreateStreamWrapper:
    """Tests for create_stream_wrapper function."""

    def test_yields_all_chunks(self) -> None:
        """Stream wrapper yields all chunks."""

        def stream() -> list[int]:
            return [1, 2, 3]

        wrapped = create_stream_wrapper(stream)
        result = list(wrapped())
        assert result == [1, 2, 3]

    def test_on_chunk_called_for_each(self) -> None:
        """on_chunk is called for each chunk."""
        chunks: list[int] = []

        def stream() -> list[int]:
            return [1, 2, 3]

        def on_chunk(chunk: int) -> None:
            chunks.append(chunk)

        wrapped = create_stream_wrapper(stream, on_chunk=on_chunk)
        list(wrapped())  # Consume iterator
        assert chunks == [1, 2, 3]

    def test_on_complete_called(self) -> None:
        """on_complete is called when stream finishes."""
        completed: list[Any] = []

        def stream() -> list[int]:
            return [1, 2, 3]

        def on_complete(final: Any) -> None:
            completed.append(final)

        wrapped = create_stream_wrapper(stream, on_complete=on_complete)
        list(wrapped())
        assert completed == [3]  # Last chunk

    def test_on_error_called(self) -> None:
        """on_error is called on exception."""
        errors: list[str] = []

        def stream() -> Any:
            yield 1

            raise ValueError("stream error")

        def on_error(e: BaseException) -> None:
            errors.append(type(e).__name__)

        wrapped = create_stream_wrapper(stream, on_error=on_error)

        try:
            list(wrapped())
        except ValueError:
            pass

        assert errors == ["ValueError"]


class TestIsVetchPatched:
    """Tests for is_vetch_patched function."""

    def test_unpatched_function(self) -> None:
        """Unpatched function returns False."""

        def func() -> None:
            pass

        assert is_vetch_patched(func) is False

    def test_patched_function(self) -> None:
        """Patched function returns True."""

        def func() -> None:
            pass

        func.vetch_patched = True  # type: ignore[attr-defined]
        assert is_vetch_patched(func) is True


class TestGetOriginal:
    """Tests for get_original function."""

    def test_unwrapped_function(self) -> None:
        """Unwrapped function returns itself."""

        def func() -> None:
            pass

        assert get_original(func) is func

    def test_wrapped_function(self) -> None:
        """Wrapped function returns original."""

        def original() -> None:
            pass

        wrapped = create_wrapper(original)
        assert get_original(wrapped) is original
