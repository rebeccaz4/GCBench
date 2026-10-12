from __future__ import annotations

import atexit
import json
import os
import shlex
import shutil
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode

from computer_env.backends.base import (
    BackgroundCommandHandle,
    BaseComputerEnvironment,
    CommandExitException,
    CommandResult,
)
from computer_env.config import EnvCreateOptions

APPTAINER_NOVNC_PORT = 6080
APPTAINER_VNC_PORT = 5900
APPTAINER_DBUS_ADDRESS = "unix:path=/tmp/runtime-user/bus"
APPTAINER_ENTRYPOINT = "/usr/local/bin/gui-synth-entrypoint.sh"
APPTAINER_ENTRYPOINT_OVERRIDE = "/tmp/gui-synth-entrypoint.sh"
LOCAL_ENTRYPOINT_PATH = (
    Path(__file__).resolve().parents[2] / "provision" / "docker" / "entrypoint.sh"
)
APPTAINER_SCREENSHOT_ATTEMPTS = 3
APPTAINER_SCREENSHOT_RETRY_DELAY_SECONDS = 1.0
APPTAINER_INSTANCE_PREFIX = "gui-synth-env"
PROCESS_SESSION_ID = f"{os.getpid()}-{uuid.uuid4().hex}"
APPTAINER_AUTO_ENTRYPOINT_READY_TIMEOUT_SECONDS = 15
APPTAINER_CLI = (
    os.getenv("APPTAINER_CLI")
    or shutil.which("apptainer")
    or shutil.which("singularity")
    or "apptainer"
)
APPTAINER_CONTAINER_ENV = {
    "DISPLAY": ":0",
    # Without an /etc/subuid entry, --fakeroot falls back to the fakeroot
    # command, under which LibreOffice cannot resolve libraries in its own
    # program directory (libreglo.so etc.).
    "LD_LIBRARY_PATH": "/usr/lib/libreoffice/program",
    "USER": "user",
    "SHELL": "/bin/bash",
    "XDG_RUNTIME_DIR": "/tmp/runtime-user",
    "DBUS_SESSION_BUS_ADDRESS": APPTAINER_DBUS_ADDRESS,
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
}

_ACTIVE_INSTANCES: set[str] = set()
_ACTIVE_INSTANCES_LOCK = threading.Lock()
_PORT_ALLOCATION_LOCK = threading.Lock()


def _apptainer_cli_env() -> dict[str, str]:
    """Keep Apptainer's host-side state out of read-only home directories."""
    env = os.environ.copy()
    state_root = Path(
        env.get("GUI_SYNTH_APPTAINER_STATE_DIR")
        or f"/tmp/gui-synth-apptainer-{os.getuid()}"
    )
    home = state_root / "home"
    cache = state_root / "cache"
    tmp = state_root / "tmp"
    for path in (home, cache, tmp):
        path.mkdir(parents=True, exist_ok=True)
    env.setdefault("APPTAINER_HOME", str(home))
    env.setdefault("APPTAINER_CACHEDIR", str(cache))
    env.setdefault("APPTAINER_TMPDIR", str(tmp))
    env["HOME"] = str(home)
    return env


def _register_active_instance(name: str) -> None:
    with _ACTIVE_INSTANCES_LOCK:
        _ACTIVE_INSTANCES.add(name)


def _unregister_active_instance(name: str) -> None:
    with _ACTIVE_INSTANCES_LOCK:
        _ACTIVE_INSTANCES.discard(name)


def _run_subprocess(
    argv: list[str],
    timeout: int | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    effective_timeout = None if timeout == 0 else timeout
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=effective_timeout,
            check=False,
            env=_apptainer_cli_env(),
        )
    except (FileNotFoundError, PermissionError) as exc:
        raise RuntimeError(
            "Apptainer CLI was not executable. Install Apptainer/Singularity, ensure "
            "`apptainer` or `singularity` is executable on PATH, or set APPTAINER_CLI "
            "to the cluster-provided executable path."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"Command timed out after {timeout}s: {' '.join(argv)}") from exc

    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Apptainer command failed")
    return result


def _run_subprocess_bytes(
    argv: list[str],
    *,
    input_bytes: bytes | None = None,
    timeout: int | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    effective_timeout = None if timeout == 0 else timeout
    try:
        result = subprocess.run(
            argv,
            input=input_bytes,
            capture_output=True,
            timeout=effective_timeout,
            check=False,
            env=_apptainer_cli_env(),
        )
    except (FileNotFoundError, PermissionError) as exc:
        raise RuntimeError(
            "Apptainer CLI was not executable. Install Apptainer/Singularity, ensure "
            "`apptainer` or `singularity` is executable on PATH, or set APPTAINER_CLI "
            "to the cluster-provided executable path."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"Command timed out after {timeout}s: {' '.join(argv)}") from exc

    if check and result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        stdout = result.stdout.decode(errors="replace").strip()
        raise RuntimeError(stderr or stdout or "Apptainer command failed")
    return result


def _find_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def _allocate_apptainer_ports() -> tuple[int, int]:
    # Apptainer instances share the host network namespace on typical HPC
    # setups, so every desktop needs distinct VNC/noVNC ports on the node.
    with _PORT_ALLOCATION_LOCK:
        vnc_port = _find_free_tcp_port()
        novnc_port = _find_free_tcp_port()
        while novnc_port == vnc_port:
            novnc_port = _find_free_tcp_port()
        return vnc_port, novnc_port


def _install_entrypoint_override(instance_name: str) -> None:
    """Copy the workspace entrypoint script into the running instance."""
    if not LOCAL_ENTRYPOINT_PATH.exists():
        return
    payload = LOCAL_ENTRYPOINT_PATH.read_bytes()
    exec_argv = [
        APPTAINER_CLI,
        "exec",
        "--env",
        "DISPLAY=:0",
        "--env",
        "USER=user",
        "--env",
        "SHELL=/bin/bash",
        "--env",
        "XDG_RUNTIME_DIR=/tmp/runtime-user",
        "--env",
        f"DBUS_SESSION_BUS_ADDRESS={APPTAINER_DBUS_ADDRESS}",
        "--env",
        "LANG=C.UTF-8",
        "--env",
        "LC_ALL=C.UTF-8",
        "--pwd",
        "/",
        f"instance://{instance_name}",
        "bash",
        "-lc",
        f"cat > {shlex.quote(APPTAINER_ENTRYPOINT_OVERRIDE)} && chmod +x {shlex.quote(APPTAINER_ENTRYPOINT_OVERRIDE)}",
    ]
    result = _run_subprocess_bytes(exec_argv, input_bytes=payload, timeout=60)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        stdout = result.stdout.decode(errors="replace").strip()
        raise RuntimeError(stderr or stdout or "Failed to install Apptainer entrypoint override")


def _instance_exists(name: str) -> bool:
    result = _run_subprocess([APPTAINER_CLI, "instance", "list", "-j"], timeout=10)
    if result.returncode != 0:
        return False
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return False
    for instance in payload.get("instances", []) or []:
        if instance.get("instance") == name:
            return True
    return False


def _stop_instance(name: str) -> bool:
    try:
        _run_subprocess([APPTAINER_CLI, "instance", "stop", "-F", name], timeout=20)
    except Exception:
        if _instance_exists(name):
            return False
    _unregister_active_instance(name)
    return True


def cleanup_active_apptainer_instances() -> list[str]:
    with _ACTIVE_INSTANCES_LOCK:
        names = set(_ACTIVE_INSTANCES)

    cleaned: list[str] = []
    for name in sorted(names):
        if _stop_instance(name):
            cleaned.append(name)
    return cleaned


def cleanup_apptainer_instances_by_metadata(metadata: dict[str, str] | None) -> list[str]:
    """Stop any of our-prefix instances whose name embeds the given run_id.

    Apptainer has no native labels/metadata. We encode `run_id` (when supplied
    by callers like evaluation.run_eval) into the instance name as
    `{APPTAINER_INSTANCE_PREFIX}-{run_id_short}-{uuid}` so we can match here.
    """

    if not metadata:
        return []

    run_id = metadata.get("run_id")
    if not run_id:
        return []
    needle = _run_id_token(run_id)

    result = _run_subprocess([APPTAINER_CLI, "instance", "list", "-j"], timeout=10)
    if result.returncode != 0:
        return []
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return []

    cleaned: list[str] = []
    for instance in payload.get("instances", []) or []:
        name = instance.get("instance") or ""
        if not name.startswith(APPTAINER_INSTANCE_PREFIX):
            continue
        if needle and needle not in name:
            continue
        if _stop_instance(name):
            cleaned.append(name)
    return cleaned


def _run_id_token(run_id: str) -> str:
    """Reduce a run_id to a short, name-safe token for embedding in instance names."""
    safe = "".join(c if c.isalnum() else "-" for c in run_id).strip("-")
    return safe[:24]


atexit.register(cleanup_active_apptainer_instances)


def _build_instance_name(run_id: str | None) -> str:
    token = _run_id_token(run_id) if run_id else ""
    suffix = uuid.uuid4().hex[:10]
    if token:
        return f"{APPTAINER_INSTANCE_PREFIX}-{token}-{suffix}"
    return f"{APPTAINER_INSTANCE_PREFIX}-{suffix}"


def _build_instance_start_argv(
    *,
    sif_path: str,
    instance_name: str,
    options: EnvCreateOptions,
    vnc_port: int,
    novnc_port: int,
) -> list[str]:
    width, height = options.resolution
    argv = [
        APPTAINER_CLI,
        "instance",
        "start",
        "--fakeroot",
        "--writable-tmpfs",
        "--containall",
        "--ipc",
        "--cleanenv",
        "--env",
        f"SCREEN_WIDTH={width}",
        "--env",
        f"SCREEN_HEIGHT={height}",
        "--env",
        f"ENV_TIMEOUT_SECONDS={options.timeout}",
        "--env",
        f"VNC_PORT={vnc_port}",
        "--env",
        f"NOVNC_PORT={novnc_port}",
    ]
    for key, value in APPTAINER_CONTAINER_ENV.items():
        argv.extend(["--env", f"{key}={value}"])
    if options.http_proxy:
        argv.extend(["--env", f"HTTP_PROXY={options.http_proxy}"])
    if options.https_proxy:
        argv.extend(["--env", f"HTTPS_PROXY={options.https_proxy}"])
    if options.no_proxy:
        argv.extend(["--env", f"NO_PROXY={options.no_proxy}"])
    argv.extend([sif_path, instance_name])
    return argv


def _tail(text: str, limit: int = 2000) -> str:
    return text[-limit:] if len(text) > limit else text


def _instance_log_paths(instance_name: str) -> tuple[Path, Path]:
    base = Path.home() / ".apptainer" / "instances" / "logs"
    # Apptainer scopes logs by hostname and user. Resolve dynamically.
    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = "localhost"
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "user"
    host_dir = base / hostname / user
    return host_dir / f"{instance_name}.out", host_dir / f"{instance_name}.err"


def _collect_instance_debug(instance_name: str) -> str:
    sections: list[str] = []
    out_path, err_path = _instance_log_paths(instance_name)
    for label, path in (("stdout", out_path), ("stderr", err_path)):
        try:
            content = path.read_text(errors="replace") if path.exists() else ""
        except Exception as exc:  # pragma: no cover - best effort diagnostics
            sections.append(f"[{label}] unavailable: {exc}")
            continue
        if content.strip():
            sections.append(f"[{label}]\n{_tail(content)}")
        else:
            sections.append(f"[{label}] empty")

    # The entrypoint subprocess writes to a host-side log file (see
    # ApptainerEnvironment._launch_entrypoint).
    host_entrypoint_log = Path(f"/tmp/{APPTAINER_INSTANCE_PREFIX}-{instance_name}.log")
    try:
        content = host_entrypoint_log.read_text(errors="replace") if host_entrypoint_log.exists() else ""
    except Exception as exc:  # pragma: no cover - best effort
        sections.append(f"[entrypoint] unavailable: {exc}")
    else:
        sections.append(f"[entrypoint]\n{_tail(content) if content.strip() else 'empty'}")
    return "\n\n".join(sections)


class ApptainerCommandsClient:
    def __init__(self, env: "ApptainerEnvironment") -> None:
        self._env = env

    def run(self, command: str, timeout: int | None = None, background: bool = False):
        wrapped = self._env._with_desktop_env(command)
        if background:
            # Keep a host-side apptainer-exec process alive as the launched
            # command's supervisor. If apptainer exec exits after `nohup ... &`,
            # GUI apps can be torn down before they create their X windows.
            exec_argv = self._env._exec_argv("bash", "-lc", wrapped)
            try:
                proc = subprocess.Popen(
                    exec_argv,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    env=_apptainer_cli_env(),
                )
            except (FileNotFoundError, PermissionError) as exc:
                raise RuntimeError(
                    "Apptainer CLI was not executable. Install Apptainer/Singularity, ensure "
                    "`apptainer` or `singularity` is executable on PATH, or set APPTAINER_CLI "
                    "to the cluster-provided executable path."
                ) from exc
            self._env._background_procs.append(proc)
            return BackgroundCommandHandle(command=command, backend="apptainer", pid=proc.pid)

        exec_argv = self._env._exec_argv("bash", "-lc", wrapped)
        result = _run_subprocess(exec_argv, timeout=timeout)
        if result.returncode != 0:
            raise CommandExitException(
                command=command,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        return CommandResult(
            command=command,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            exit_code=result.returncode,
        )


class ApptainerFilesClient:
    def __init__(self, env: "ApptainerEnvironment") -> None:
        self._env = env

    def write(self, path: str, data) -> None:
        payload = data.encode() if isinstance(data, str) else data
        parent = str(Path(path).parent)
        self._env.commands.run(f"mkdir -p {shlex.quote(parent)}", timeout=10)

        exec_argv = self._env._exec_argv("bash", "-lc", f"cat > {shlex.quote(path)}")
        write_result = _run_subprocess_bytes(exec_argv, input_bytes=payload, timeout=60)
        if write_result.returncode != 0:
            raise RuntimeError(
                write_result.stderr.decode(errors="replace").strip()
                or write_result.stdout.decode(errors="replace").strip()
                or "Failed to write file via apptainer exec"
            )
        # Mirror docker backend: chown to user when possible. Under fakeroot we
        # are uid=0 inside, so we attempt the chown but swallow failures.
        self._env.commands.run(
            f"chown user:user {shlex.quote(path)} 2>/dev/null || true",
            timeout=10,
        )

    def read(self, path: str, format: str = "text"):
        exec_argv = self._env._exec_argv("bash", "-lc", f"cat {shlex.quote(path)}")
        result = _run_subprocess_bytes(exec_argv, timeout=60)
        if result.returncode != 0:
            raise FileNotFoundError(path)
        data = result.stdout
        if format in {"bytes", "binary"}:
            return data
        return data.decode()

    def remove(self, path: str) -> None:
        self._env.commands.run(f"rm -f {shlex.quote(path)}", timeout=10)


class ApptainerStreamClient:
    def __init__(self, env: "ApptainerEnvironment") -> None:
        self._env = env

    def start(self, **kwargs):
        self._env.wait_ready()

    def get_url(
        self,
        resize: str | None = None,
        view_only: bool = False,
        auth_key: str | None = None,
    ):
        params = {"autoconnect": "true"}
        if resize:
            params["resize"] = resize
        if view_only:
            params["view_only"] = "true"
        if auth_key:
            params["password"] = auth_key
        query = urlencode(params)
        return f"http://127.0.0.1:{self._env.stream_port}/vnc.html?{query}"

    def stop(self):
        return None


class ApptainerEnvironment(BaseComputerEnvironment):
    backend_name = "apptainer"

    def __init__(
        self,
        *,
        instance_name: str,
        sif_path: str,
        stream_port: int,
        vnc_port: int,
        timeout: int,
        resolution: tuple[int, int],
        ready_timeout: int,
    ) -> None:
        self.instance_name = instance_name
        self.sif_path = sif_path
        self.stream_port = stream_port
        self.vnc_port = vnc_port
        self.timeout = timeout
        self.resolution = resolution
        self.ready_timeout = ready_timeout
        self.commands = ApptainerCommandsClient(self)
        self.files = ApptainerFilesClient(self)
        self.stream = ApptainerStreamClient(self)
        self._killed = False
        self._entrypoint_proc: subprocess.Popen | None = None
        self._entrypoint_log = None
        self._background_procs: list[subprocess.Popen] = []

    @classmethod
    def create(cls, options: EnvCreateOptions) -> "ApptainerEnvironment":
        sif_path = options.apptainer_image
        if not sif_path or not os.path.exists(sif_path):
            raise RuntimeError(
                f"Apptainer SIF image not found at '{sif_path}'. Pull it first with:\n"
                f"  bash computer_env/provision/apptainer/pull_image.sh "
                f"docker://<image:tag> {sif_path or '<output.sif>'}"
            )

        run_id_metadata = (options.apptainer_metadata or {}).get("run_id")
        instance_name = _build_instance_name(run_id_metadata)
        vnc_port, novnc_port = _allocate_apptainer_ports()
        start_argv = _build_instance_start_argv(
            sif_path=sif_path,
            instance_name=instance_name,
            options=options,
            vnc_port=vnc_port,
            novnc_port=novnc_port,
        )
        start_result = _run_subprocess(start_argv, timeout=60)
        if start_result.returncode != 0:
            raise RuntimeError(
                (start_result.stderr or start_result.stdout or "apptainer instance start failed").strip()
            )
        _register_active_instance(instance_name)
        _install_entrypoint_override(instance_name)

        auto_env = cls(
            instance_name=instance_name,
            sif_path=sif_path,
            stream_port=novnc_port,
            vnc_port=vnc_port,
            timeout=options.timeout,
            resolution=options.resolution,
            ready_timeout=min(
                options.apptainer_ready_timeout,
                APPTAINER_AUTO_ENTRYPOINT_READY_TIMEOUT_SECONDS,
            ),
        )
        try:
            auto_env.wait_ready()
        except Exception:
            pass
        else:
            auto_env.ready_timeout = options.apptainer_ready_timeout
            return auto_env

        entrypoint_proc: subprocess.Popen | None = None
        entrypoint_log = None
        try:
            entrypoint_proc, entrypoint_log = cls._launch_entrypoint(
                instance_name,
                vnc_port=vnc_port,
                novnc_port=novnc_port,
            )
            env = cls(
                instance_name=instance_name,
                sif_path=sif_path,
                stream_port=novnc_port,
                vnc_port=vnc_port,
                timeout=options.timeout,
                resolution=options.resolution,
                ready_timeout=options.apptainer_ready_timeout,
            )
            env._entrypoint_proc = entrypoint_proc
            env._entrypoint_log = entrypoint_log
            env.wait_ready()
            return env
        except BaseException:
            if entrypoint_proc is not None:
                try:
                    entrypoint_proc.terminate()
                except Exception:
                    pass
            if entrypoint_log is not None:
                try:
                    entrypoint_log.close()
                except Exception:
                    pass
            try:
                _run_subprocess([APPTAINER_CLI, "instance", "stop", "-F", instance_name], timeout=20)
            except Exception:
                pass
            finally:
                _unregister_active_instance(instance_name)
            raise

    @staticmethod
    def _launch_entrypoint(instance_name: str, *, vnc_port: int, novnc_port: int):
        # Apptainer doesn't auto-run the OCI ENTRYPOINT for instances, and
        # backgrounding inside `apptainer exec bash -c 'nohup ... &'` hangs the
        # entrypoint's first mkdir reliably. Instead, launch it as a foreground
        # apptainer-exec subprocess that stays alive as the entrypoint's
        # supervisor; `apptainer instance stop` later tears the whole tree down.
        log_path = f"/tmp/{APPTAINER_INSTANCE_PREFIX}-{instance_name}.log"
        log_handle = open(log_path, "wb")
        try:
            container_env_args = [
                part
                for key, value in {
                    **APPTAINER_CONTAINER_ENV,
                    "VNC_PORT": str(vnc_port),
                    "NOVNC_PORT": str(novnc_port),
                }.items()
                for part in ("--env", f"{key}={value}")
            ]
            proc = subprocess.Popen(
                [
                    APPTAINER_CLI,
                    "exec",
                    *container_env_args,
                    "--pwd",
                    "/",
                    f"instance://{instance_name}",
                    "bash",
                    "-lc",
                    (
                        "export HOME=/home/user; "
                        "export USER=user; "
                        "export SHELL=/bin/bash; "
                        "export XDG_RUNTIME_DIR=/tmp/runtime-user; "
                        f"export VNC_PORT={int(vnc_port)}; "
                        f"export NOVNC_PORT={int(novnc_port)}; "
                        f"export DBUS_SESSION_BUS_ADDRESS={shlex.quote(APPTAINER_DBUS_ADDRESS)}; "
                        f"if [ -x {shlex.quote(APPTAINER_ENTRYPOINT_OVERRIDE)} ]; then "
                        f"exec {shlex.quote(APPTAINER_ENTRYPOINT_OVERRIDE)}; "
                        f"else exec {shlex.quote(APPTAINER_ENTRYPOINT)}; fi"
                    ),
                ],
                stdout=log_handle,
                stderr=log_handle,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=_apptainer_cli_env(),
            )
        except Exception:
            log_handle.close()
            raise
        return proc, log_handle

    def _exec_argv(self, *cmd: str, stdin: bool = False) -> list[str]:
        argv = [APPTAINER_CLI, "exec"]
        for key, value in APPTAINER_CONTAINER_ENV.items():
            argv.extend(["--env", f"{key}={value}"])
        argv.extend(["--pwd", "/", f"instance://{self.instance_name}", *cmd])
        return argv

    def _with_desktop_env(self, command: str) -> str:
        return (
            "export DISPLAY=:0; "
            "export HOME=/home/user; "
            "export USER=user; "
            "export SHELL=/bin/bash; "
            "export XDG_RUNTIME_DIR=/tmp/runtime-user; "
            f"export DBUS_SESSION_BUS_ADDRESS=${{DBUS_SESSION_BUS_ADDRESS:-{shlex.quote(APPTAINER_DBUS_ADDRESS)}}}; "
            f"{command}"
        )

    def wait_ready(self) -> None:
        deadline = time.time() + self.ready_timeout
        last_error = "desktop not ready"
        while time.time() < deadline:
            if not _instance_exists(self.instance_name):
                debug_info = _collect_instance_debug(self.instance_name)
                raise RuntimeError(
                    "Apptainer instance exited before becoming ready.\n"
                    f"{debug_info}"
                )

            try:
                check = _run_subprocess(
                    self._exec_argv(
                        "bash",
                        "-lc",
                        self._with_desktop_env(
                            "DISPLAY=:0 xset q >/dev/null 2>&1 && "
                            "pgrep -f x11vnc >/dev/null 2>&1 && "
                            "pgrep -f websockify >/dev/null 2>&1"
                        ),
                    ),
                    timeout=10,
                )
            except TimeoutError:
                last_error = "ready check apptainer exec timed out"
                time.sleep(1)
                continue
            if check.returncode == 0:
                try:
                    with socket.create_connection(("127.0.0.1", self.stream_port), timeout=1):
                        return
                except OSError as exc:
                    last_error = str(exc)
            else:
                last_error = (check.stderr or check.stdout or last_error).strip()

            time.sleep(1)

        debug_info = _collect_instance_debug(self.instance_name)
        raise RuntimeError(
            "Apptainer desktop environment did not become ready.\n"
            f"reason: {last_error}\n\n{debug_info}"
        )

    def screenshot(self):
        last_error: Exception | None = None

        for attempt in range(1, APPTAINER_SCREENSHOT_ATTEMPTS + 1):
            screenshot_path = f"/tmp/gui_synth_env_screenshot_{uuid.uuid4().hex}.png"
            try:
                self.commands.run(
                    f"DISPLAY=:0 scrot --pointer {shlex.quote(screenshot_path)}",
                    timeout=20,
                )
                data = self.files.read(screenshot_path, format="bytes")
                self.commands.run(f"rm -f {shlex.quote(screenshot_path)}", timeout=10)
                return data
            except Exception as exc:
                last_error = exc
                try:
                    self.commands.run(f"rm -f {shlex.quote(screenshot_path)}", timeout=10)
                except Exception:
                    pass
                if attempt < APPTAINER_SCREENSHOT_ATTEMPTS:
                    time.sleep(APPTAINER_SCREENSHOT_RETRY_DELAY_SECONDS)

        raise RuntimeError(
            "Apptainer screenshot failed after "
            f"{APPTAINER_SCREENSHOT_ATTEMPTS} attempts: {last_error}"
        )

    def kill(self) -> None:
        if self._killed:
            return
        self._killed = True
        try:
            _run_subprocess([APPTAINER_CLI, "instance", "stop", "-F", self.instance_name], timeout=20)
        finally:
            if self._entrypoint_proc is not None:
                try:
                    self._entrypoint_proc.wait(timeout=10)
                except Exception:
                    try:
                        self._entrypoint_proc.kill()
                    except Exception:
                        pass
            for proc in self._background_procs:
                if proc.poll() is not None:
                    continue
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            if self._entrypoint_log is not None:
                try:
                    self._entrypoint_log.close()
                except Exception:
                    pass
            _unregister_active_instance(self.instance_name)


def create_apptainer_env(options: EnvCreateOptions) -> ApptainerEnvironment:
    return ApptainerEnvironment.create(options)
