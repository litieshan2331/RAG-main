from app.core.errors import to_http_exception


def test_runtime_error_maps_to_bad_request() -> None:
    http_error = to_http_exception(RuntimeError("bad state"))

    assert http_error.status_code == 400
    assert http_error.detail == "bad state"


def test_unknown_error_maps_to_internal_error() -> None:
    http_error = to_http_exception(Exception("unknown"))

    assert http_error.status_code == 500
