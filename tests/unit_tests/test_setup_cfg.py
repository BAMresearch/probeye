# standard library
import os

# local imports
from probeye._setup_cfg import version_constraint_free_packages
from probeye._setup_cfg import version_constraint_free_dependencies


def test_version_removal():
    p_in_list = [
        "numpy<2",
        "scipy!=1.10, >1.0",
        "pandas",
        "torch[dev]",
        "matplotlib[dev]<=2",
        "tensorflow==1.23rc",
    ]
    expected_p_out_list = [
        "numpy",
        "scipy",
        "pandas",
        "torch[dev]",
        "matplotlib[dev]",
        "tensorflow",
    ]
    for ii in range(len(p_in_list)):
        p_in_list_ii = p_in_list[: (ii + 1)]
        expected_p_out_list_ii = expected_p_out_list[: (ii + 1)]

        # .......................................................
        # Pure dangling list
        # .......................................................
        p_in = "\n" + "\n".join(p_in_list_ii)
        p_out_list_ii = version_constraint_free_packages(setup_cfg_packages=p_in)
        assert p_out_list_ii == expected_p_out_list_ii

        # with extra space
        p_in = "\n" + "\n ".join(p_in_list_ii)
        p_out_list_ii = version_constraint_free_packages(setup_cfg_packages=p_in)
        assert p_out_list_ii == expected_p_out_list_ii

        # .......................................................
        # Pure semi-colon separated list
        # .......................................................
        p_in = ";".join(p_in_list_ii)
        p_out_list_ii = version_constraint_free_packages(setup_cfg_packages=p_in)
        assert p_out_list_ii == expected_p_out_list_ii

        # .......................................................
        # Dangling list combined with semi-colon separated list
        # .......................................................
        if ii > 2:
            p_in = (
                "\n" + "\n".join(p_in_list_ii[:2]) + "\n" + ";".join(p_in_list_ii[2:])
            )
            p_out_list_ii = version_constraint_free_packages(setup_cfg_packages=p_in)
            assert p_out_list_ii == expected_p_out_list_ii

            # reversed order
            p_in = "\n".join(p_in_list_ii[:2]) + "\n" + ";".join(p_in_list_ii[2:])
            p_out_list_ii = version_constraint_free_packages(setup_cfg_packages=p_in)
            assert p_out_list_ii == expected_p_out_list_ii


def test_version_constraint_free_dependencies():
    dir_path = os.path.dirname(__file__)
    setup_cfg = os.path.join(dir_path, "../../setup.cfg")
    version_constraint_free_dependencies(
        options_field="install_requires", test=True, setup_cfg=setup_cfg
    )
