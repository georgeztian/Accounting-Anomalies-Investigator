from ffi._env_guard import require_virtualenv

require_virtualenv()

from ffi.cli import main  # noqa: E402

main()
