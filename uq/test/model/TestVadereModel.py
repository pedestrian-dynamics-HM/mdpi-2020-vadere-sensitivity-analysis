import unittest
import numpy as np
from numpy.random import RandomState
import os
from uq.utils.model_function import VadereModel

KEY = ["attributesPedestrian.speedDistributionMean", "attributesPedestrian.speedDistributionStandardDeviation",
       "sources.[id==3].spawnNumber",
       "sources.[id==3].distributionParameters", "obstacles.[id==1].y",
       "obstacleRepulsionMaxWeight"]  # uncertain parameters

TEST_INPUT = np.array([[1.34], [0.26], [180], [1], [8.5], [0.5]])  # legal test input
QOI = "mean_density.txt"  # quantity of interest
RUN_LOCAL = True  # run on local machine (vs. run on server)

cur_dir = os.path.dirname(os.path.realpath(__file__))
PATH2SCENARIOS = os.path.abspath(os.path.join(cur_dir, "../../scenarios"))

PATH2MODEL = os.path.join(PATH2SCENARIOS, "vadere-console.jar")
PATH2SCENARIO_FIXED_SEED = os.path.join(PATH2SCENARIOS, "Liddle_bhm_v3_fixed_seed.scenario")

NJOBS = -1
LOG_LEVEL = "OFF"


class TestVadereModel(unittest.TestCase):
    def test_model_parameters(self):
        # test_model = TestModel()
        test_model = VadereModel(RUN_LOCAL, PATH2SCENARIO_FIXED_SEED, PATH2MODEL, KEY, QOI, NJOBS, LOG_LEVEL)

        self.assertEqual(test_model.get_dimension(), len(KEY))
        self.assertEqual(test_model.get_path_2_model(), PATH2MODEL)
        self.assertEqual(test_model.get_path_2_scenario(), PATH2SCENARIO_FIXED_SEED)
        self.assertEqual(test_model.get_key(), KEY)
        self.assertEqual(test_model.get_qoi(), QOI)
        self.assertEqual(test_model.get_loglvl(), LOG_LEVEL)
        self.assertEqual(test_model.get_n_jobs(), NJOBS)

    def test_eval_model_two_runs_simple_fixed_seed_osm(self):
        path_fixed_seed_osm = os.path.join(PATH2SCENARIOS, "Liddle_osm_v3_fixed_seed.scenario")
        key_osm = ["queueWidthLoading"]
        test_input_osm = np.array([[1.0]])
        test_model = VadereModel(RUN_LOCAL, path_fixed_seed_osm, PATH2MODEL, key_osm, QOI, NJOBS, LOG_LEVEL)

        result_single_1 = test_model.eval_model(test_input_osm, bool_fixed_seed=True)
        result_single_2 = test_model.eval_model(test_input_osm, bool_fixed_seed=True)

        np.testing.assert_almost_equal(result_single_1, result_single_2)

    def test_eval_model_two_runs_simple_free_seed_osm(self):
        path_fixed_seed_osm = os.path.join(PATH2SCENARIOS, "Liddle_osm_v3_fixed_seed.scenario")
        key_osm = ["queueWidthLoading"]
        test_input_osm = np.array([[1.0]])
        seed = 634679326
        test_model = VadereModel(RUN_LOCAL, path_fixed_seed_osm, PATH2MODEL, key_osm, QOI, NJOBS, LOG_LEVEL)

        result_single_1 = test_model.eval_model(test_input_osm, bool_fixed_seed=False, random_state=RandomState(seed))
        result_single_2 = test_model.eval_model(test_input_osm, bool_fixed_seed=False, random_state=RandomState(seed))

        np.testing.assert_almost_equal(result_single_1, result_single_2)

    def test_eval_model_two_runs_fixed_seed_osm(self):
        path_fixed_seed_osm = os.path.join(PATH2SCENARIOS, "Liddle_osm_v3_fixed_seed.scenario")

        key_osm = ["attributesPedestrian.speedDistributionMean",
                   "attributesPedestrian.speedDistributionStandardDeviation",
                   "sources.[id==3].spawnNumber",
                   "sources.[id==3].distributionParameters", "obstacles.[id==1].y"]  # uncertain parameters

        test_input_osm = np.array([[1.34], [0.26], [180], [1], [8.5]])  # legal test input
        test_model = VadereModel(RUN_LOCAL, path_fixed_seed_osm, PATH2MODEL, key_osm, QOI, NJOBS, LOG_LEVEL)

        result_single_1 = test_model.eval_model(test_input_osm, bool_fixed_seed=True)
        result_single_2 = test_model.eval_model(test_input_osm, bool_fixed_seed=True)

        np.testing.assert_almost_equal(result_single_1, result_single_2)

    def test_eval_model_averaged_vadere_internal_seed(self):
        test_model = VadereModel(RUN_LOCAL, PATH2SCENARIO_FIXED_SEED, PATH2MODEL, KEY, QOI, NJOBS, LOG_LEVEL)
        # assure that default value passing works
        result_single = test_model.eval_model(TEST_INPUT, bool_fixed_seed=True, random_state=None)
        single_output, _, _ = test_model.eval_model_averaged(TEST_INPUT, 1, bool_fixed_seed=True, random_state=None)
        single_output_averaged, _, _ = test_model.eval_model_averaged(TEST_INPUT, 3, bool_fixed_seed=True,
                                                                      random_state=None)

        np.testing.assert_almost_equal(result_single, single_output)
        np.testing.assert_almost_equal(single_output_averaged, single_output)

        # assure that averaging gives the same results as single run for fixed seed
        result_averaged, _, _ = test_model.eval_model_averaged(TEST_INPUT, 3, bool_fixed_seed=True)
        np.testing.assert_almost_equal(result_single, result_averaged)

    def test_eval_model_averaged_vadere_external_seed(self):
        # assure that averaging with the same random seeds works
        seed = 39636

        test_model = VadereModel(RUN_LOCAL, PATH2SCENARIO_FIXED_SEED, PATH2MODEL, KEY, QOI, NJOBS, LOG_LEVEL)

        result_averaged1, input1, result1 = test_model.eval_model_averaged(TEST_INPUT, 3, bool_fixed_seed=False,
                                                                      random_state=RandomState(seed))

        result_averaged2, input2, result2 = test_model.eval_model_averaged(TEST_INPUT, 3, bool_fixed_seed=False,
                                                                      random_state=RandomState(seed))

        # make sure the generated inputs are identical
        np.testing.assert_array_almost_equal(input1, input2)

        # make sure the individual results are identical
        np.testing.assert_array_almost_equal(result1, result2)

        # make sure the averaged results are identical
        np.testing.assert_allclose(result_averaged1, result_averaged2)

        np.testing.assert_allclose(1,1)

if __name__ == '__main__':
    unittest.main()
