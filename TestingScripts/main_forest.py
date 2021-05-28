
"""Test Functions"""
def test_nav():
    env = ForestSim(map_name)
    vehicle = NavTestVehicle(nav_name, env.sim_conf)

    test_single_vehicle(env, vehicle, True, 100, wait=False)
    # test_single_vehicle(env, vehicle, True, 1, add_obs=False, wait=False)



def test_follow_the_gap():
    sim_conf = lib.load_conf("fgm_config")
    env = ForestSim(map_name, sim_conf)
    vehicle = ForestFGM()

    # test_single_vehicle(env, vehicle, True, 10, False, vis=True)
    test_single_vehicle(env, vehicle, True, 100, add_obs=True, vis=False)


def test_oracle():
    env = ForestSim(map_name)
    vehicle = Oracle(env.sim_conf)

    test_oracle_forest(env, vehicle, True, 100, True, wait=False)
    # test_oracle_forest(env, vehicle, True, 1, False, wait=False)


def test_mod():
    env = ForestSim(map_name)
    vehicle = ModVehicleTest(mod_name, map_name, env.sim_conf)
    # vehicle = ModVehicleTest("ModForest_nr6", map_name, env.sim_conf)

    test_single_vehicle(env, vehicle, True, 100, wait=False, vis=False)
    # test_single_vehicle(env, vehicle, False, 100, wait=False, vis=False)
    # test_single_vehicle(env, vehicle, True, 1, add_obs=False, wait=False, vis=False)




def big_test():
    env = ForestSim(map_name)
    test = TestVehicles(env.sim_conf, eval_name)

    vehicle = NavTestVehicle(nav_name, env.sim_conf)
    test.add_vehicle(vehicle)

    vehicle = ForestFGM()
    test.add_vehicle(vehicle)

    vehicle = Oracle(env.sim_conf)
    test.add_vehicle(vehicle)

    vehicle = ModVehicleTest(mod_name, map_name, env.sim_conf)
    test.add_vehicle(vehicle)

    # test.run_eval(env, 1, True)
    test.run_eval(env, 100, False, wait=False)



def test_repeat():
    env = ForestSim(map_name)
    test = TestVehicles(env.sim_conf, repeat_name)

    for i in range(10):
        train_name = f"ModRepeat_forest_{i}"
        vehicle = ModVehicleTest(train_name, map_name, env.sim_conf)
        test.add_vehicle(vehicle)

    # test.run_eval(env, 1000, False)
    test.run_eval(env, 100, False)



if __name__ == "__main__":
    
    # train_nav()
    # train_mod()
    # train_repeatability()

    test_nav()
    # test_follow_the_gap()
    # test_oracle()
    # test_mod()
    # test_repeat()
# 
    # big_test()







