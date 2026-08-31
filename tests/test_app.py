from app import app


def test_public_routes_exclude_fastapi_documentation_endpoints():
    routes = {route.path for route in app.routes}

    assert "/ping" in routes
    assert "/rank" in routes
    assert "/docs" not in routes
    assert "/redoc" not in routes
    assert "/openapi.json" not in routes
    assert app.docs_url is None
    assert app.redoc_url is None
    assert app.openapi_url is None
