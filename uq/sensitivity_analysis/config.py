import os

from uq.utils.model_function import VadereModel
from uq.utils.datatype import get_dimension


def configure_vadere_sa(run_local: bool, scenario_name: str, key, qoi):
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    path2output = os.getcwd()
    print(path2output)

    scenario_path = os.path.join(cur_dir, "../scenarios/")
    path2model = os.path.abspath(os.path.join(scenario_path, "vadere-console.jar"))
    path2scenario = os.path.abspath(os.path.join(scenario_path, scenario_name))
    print(path2model)
    print(path2scenario)

    # test_model = TestModel()
    test_model = VadereModel(run_local, path2scenario, path2model, key, qoi, n_jobs=-1, log_lvl="OFF")

    m = get_dimension(key)

    return test_model, m, path2output
