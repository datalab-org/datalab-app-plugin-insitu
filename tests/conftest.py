def pytest_addoption(parser):
    parser.addoption(
        "--show-plots", action="store_true", default=False, help="Show plots made during tests"
    )
