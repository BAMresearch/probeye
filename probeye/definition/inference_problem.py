# standard library
from typing import Optional


class InferenceProblem:
    def __init__(
        self,
        name: str,
        use_default_logger: bool = True,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        print_header: bool = True,
    ):
        self.name = name
        self.use_default_logger = use_default_logger
        self.log_level = log_level
        self.log_file = log_file
        self.print_header = print_header

        raise NotImplementedError(
            f"=================================================================\n"
            f"In version 2.1.5 the main class 'InferenceProblem' was renamed to "
            f"'InverseProblem'.\nIn order to adjust your script to this change you "
            f"just have to change your import from\n"
            f"  from probeye.definition.inference_problem import InferenceProblem to\n"
            f"  from probeye.definition.inverse_problem import InverseProblem\n"
            f"and then use the class 'InverseProblem' instead of 'InferenceProblem'.\n"
            f"========================================================================="
            f"============="
        )
