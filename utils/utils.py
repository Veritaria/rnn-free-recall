import importlib
import pathlib
import matplotlib.pyplot as plt


def import_attr(module_and_attr_name):
    """
    Import an attr (class or function) from a (python) module, e.g. 'models.RNN' (taken from the Full Stack DL course)
    Args:
        module_and_attr_name: if list of str, try to import each attr in order and return the first that can be imported
    """
    module_name, class_name = module_and_attr_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    attr_ = getattr(module, class_name)

    return attr_


def savefig(save_dir, filename, format='png', close=True):
    """
    format: 'png' | 'pdf' | 'svg'
    """
    if save_dir is None:
        return None
    if isinstance(save_dir, str):
        save_dir = pathlib.Path(save_dir)

    filename = filename.replace('.', '-')
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    plt.savefig(save_dir/(filename + "." + format))
    # if format != 'png':
    #     plt.savefig(save_dir/(filename + ".png"))
    if close:
        plt.close()
