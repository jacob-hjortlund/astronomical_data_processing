import paths
import numpy as np
import pandas as pd
import uncertainties as unc

from pathlib import Path


def convert_variable_to_latex(
    variable: float,
    error: float = None,
    sigfigs: int = None,
    decimals: int = None,
    units: str = None,
) -> str:
    """
    Round a variable with potential uncertainty to either sigfigs significant figures or
    decimals decimal places and return it as a LaTeX-formatted string.

    Args:
        variable (float): The variable to round.
        error (float, optional): The error on the variable. Defaults to None.
        sigfigs (int, optional): Number of significant figures to round to. Defaults to None.
                                    If None, then decimals must be provided.
        decimals (int, optional): Number of decimal places to round to. Defaults to None.
                                    If None, then sigfigs must be provided.
        units (str, optional): Units of the variable. Defaults to None.

    Returns:
        str: The rounded variable as a LaTeX-formatted string.
    """

    if sigfigs is None and decimals is None:
        raise ValueError("Must provide either sigfigs or decimals")
    if sigfigs is not None and decimals is not None:
        raise ValueError("Cannot provide both sigfigs and decimals")
    if decimals is not None and error is not None:
        raise ValueError("Cannot provide both decimals and error")

    if decimals is not None:
        n_digits = len(str(variable).split(".")[0])
        sigfigs = decimals + n_digits
        _error = variable
    elif error is None:
        _error = variable
    else:
        _error = error

    variable = unc.ufloat(variable, _error)
    rounded_variable = variable.__format__(f".{sigfigs}uL")

    if error is None or decimals is not None:
        rounded_variable_list = rounded_variable.split(" ")
        if "10^" in rounded_variable:
            rounded_variable = (
                rounded_variable_list[0]
                + rounded_variable_list[-2]
                + rounded_variable_list[-1]
            )
            rounded_variable = rounded_variable.replace("\\left(", "")
            rounded_variable = rounded_variable.replace("\\right)", "")
        else:
            rounded_variable = rounded_variable_list[0]

    if units is not None:
        rounded_variable = rounded_variable + " " + units

    return rounded_variable


def save_variable_to_latex(
    variable: float,
    variable_error: float = None,
    variable_name: str = "var",
    variable_units: str = None,
    filename: str = None,
    sigfigs: int = None,
    decimals: int = None,
    path: str = None,
    mode: str = "a",
):
    """
    Save a number with potential uncertainty to a file in a dict-like format
    of the form

        variable_name: variable\\n

    where variable is rounded to either sigfigs significant figures or decimal decimal places.

    Args:
        variable (float): The variable / number to save.
        variable_error (float, optional): Error on the variable. Defaults to None.
        variable_name (str, optional): Name of the variable. Defaults to 'Var'.
        variable_units (str, optional): Units of the variable. Defaults to None.
        filename (str, optional): Name of the file to save to. Defaults to variable_name.
        sigfigs (int, optional): Number of significant figures to round to. Defaults to None.
                                 If None, then decimals must be provided.
        decimals (int, optional): Number of decimal places to round to. Defaults to None.
                                  If None, then sigfigs must be provided.
        path (str, optional): Path to save the file to. Defaults to paths.output.
        mode (str, optional): Mode to open the file in. Defaults to 'a'.
    """

    if path is None:
        path = paths.output
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / filename

    variable_str = convert_variable_to_latex(
        variable=variable,
        error=variable_error,
        sigfigs=sigfigs,
        decimals=decimals,
        units=variable_units,
    )

    with open(file_path, mode) as f:
        f.write(f"{variable_name},{variable_str}")
