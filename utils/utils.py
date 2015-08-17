def validate_identifier(identifier, must_exist=True):
    """Check whether a used identifier is a valid one or raise an error.
    
    Optionally also check if there is already an experiment with the identifier
    and raise an error if there is none yet.
    
    Valid identifiers contain only:
        a-z
        A-Z
        0-9
        _
    
    Args:
        identifier: Identifier to check for validity.
        must_exist: If set to true and no experiment uses the identifier yet,
            an error will be raised.
    
    Returns:
        void
    """
    if not identifier or identifier != re.sub("[^a-zA-Z0-9_]", "", identifier):
        raise Exception("Invalid characters in identifier, only a-z A-Z 0-9 and _ are allowed.")
    if must_exist:
        if not identifier_exists(identifier):
            raise Exception("No model with identifier '{}' seems to exist.".format(identifier))

def identifier_exists(identifier):
    """Returns True if the provided identifier exists.
    The existence and check by checking if there is a history (csv file)
    with the provided identifier.
    
    Args:
        identifier: Identifier of the experiment.

    Returns:
        True if an experiment with the identifier exists.
        False otherwise.
    """
    filepath = SAVE_CSV_FILEPATH.format(identifier=identifier)
    if os.path.isfile(filepath):
        return True
    else:
        return False

def ask_continue(message):
    """Displays the message and waits for a "y" (yes) or "n" (no) input by the user.
    
    Args:
        message: The message to display.

    Returns:
        True if the user has entered "y" (for yes).
        False if the user has entered "n" (for no).
    """
    choice = raw_input(message)
    while choice not in ["y", "n"]:
        choice = raw_input("Enter 'y' (yes) or 'n' (no) to continue.")
    return choice == "y"
