"""Shared bits for the benchmark scripts."""

import platform
import subprocess


def machine() -> str:
    """The CPU, as readable as the OS will tell us: "Apple M3", "AMD EPYC 7763 ..."."""
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, check=True)  # fmt: skip
            return out.stdout.strip()
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.machine()
