"""Keep GPU computation separate from the persistent workflow process."""

try:
    from spaces import GPU as gpu_task
except ImportError:
    def gpu_task(*_args, **_kwargs):
        """On a local CPU/GPU machine, inference runs in the current process."""
        return lambda function: function

