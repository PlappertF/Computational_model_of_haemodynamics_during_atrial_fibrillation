import src.init


class CardioGrowth(object):
    """
    Main model class, initializing and linking all main functions
    """

    def __init__(self, model_pars):
        src.init.init(self, model_pars=model_pars)

    from src.init import initialize_multipatch, initialize_batch
    from src.check_input import check_input_params
    from src.import_export import import_pars
    from src.utils import change_pars
    from src.rhythm import rhythm_is_a_dancer
    from src.activation import initialize_activations
    from src.avn import get_rhythm_parameters, run_avn_model
    from src.post_processing import do_all_post_processing