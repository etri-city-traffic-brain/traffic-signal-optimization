import argparse
import os
import sys
import numpy as np
import itertools

from sumolib import checkBinary

from rl.agents.dqn import Learner, get_state_1d
import traci

from collections import deque
np.set_printoptions(threshold=sys.maxsize)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='test')
parser.add_argument('--model-num', type=str, default='0')
parser.add_argument('--disp', choices=['y', 'n'], default='n')
args = parser.parse_args()

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

if args.mode =='train':
    sumoConfig = "env/3x3/in-out.sumocfg"
elif args.mode == 'test':
    sumoConfig = "env/3x3/in-out-test.sumocfg"

action_space_size = 21
agent = Learner(action_space_size, 1, '1d')

epochs = 200000

if args.mode == 'train':
    if args.disp == 'y':
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')
    sumoCmd = [sumoBinary, "-c", sumoConfig, "--start", "--time-to-teleport", "-1"]

    for simulation in range(epochs):
        traci.start(sumoCmd)
        simulationSteps = 0

        control_tl = 'B2'
        state_tl_lsit = ['B2','B1']
        target_links = ['B2B1', 'B1B0']

        total_reward = 0

        done = False
        tmp_hn = 0
        tpnumber = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            state = get_state_1d(state_tl_lsit)
            action = agent.act(state)

            tl_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(control_tl)
            phase1_duration = 20 + action*2
            phase2_duration = 90-3-3-phase1_duration
            tl_logic[0].phases[0].duration = phase1_duration
            tl_logic[0].phases[2].duration = phase2_duration
            
            traci.trafficlight.setProgramLogic(control_tl, tl_logic[0])
            
            reward_list = []
            for i in range(90):
                traci.simulationStep()
                tpnumber += traci.simulation.getStartingTeleportNumber()
                simulationSteps += 1
                reward_tmp = -(traci.edge.getLastStepVehicleNumber(target_links[0]) - traci.edge.getLastStepVehicleNumber(target_links[1]))
                reward_list = np.append(reward_list, reward_tmp)

            reward = np.mean(reward_list) + phase1_duration*0.1

            total_reward += reward

            next_state = get_state_1d(state_tl_lsit)

            if simulationSteps>10000:
                done = True
                agent.remember(state, action, reward, next_state, done)
                break
            
            agent.remember(state, action, reward, next_state, done)

            if simulationSteps % 900 == 0 and agent.batch_size < len(agent.memory):
                agent.increase_target_update_counter()
                agent.replay()


        print(simulationSteps)

        traci.close()
        with open("ResultsOfSimulations_singleAgent_inout.txt", "a") as f:
            f.write("Simulation {}: total_reward {} simulationSteps {} tpnumber {}\n".format(simulation, total_reward, simulationSteps, tpnumber))
            if simulation % 20 == 0:
                agent.save("singleAgent_inout/traffic_epoch{}".format(simulation))


elif args.mode == 'test':
    agent = Learner(action_space_size, 0, '1d')

    sumoBinary = checkBinary('sumo-gui')
    sumoCmd = [sumoBinary, "-c", sumoConfig, "--start", "--device.rerouting.probability", "0"]

    model_num = args.model_num
    
    agent.load("singleAgent_inout/traffic_epoch{}".format(model_num))
    
    traci.start(sumoCmd)
    simulationSteps = 0

    control_tl = 'B2'
    state_tl_lsit = ['B2','B1']
    target_links = ['B2B1', 'B1B0']

    total_reward = 0
    total_reward_withaction = 0

    done = False
    tmp_hn = 0
    tpnumber = 0
    arrived_arr = []
    while traci.simulation.getMinExpectedNumber() > 0:
        state = get_state_1d(state_tl_lsit)
        action = agent.act(state)

        tl_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(control_tl)
        phase1_duration = 20 + action*2
        phase2_duration = 90-3-3-phase1_duration
        tl_logic[0].phases[0].duration = phase1_duration
        tl_logic[0].phases[2].duration = phase2_duration
        print('p1 duration : {} p2 duration : {}'.format(phase1_duration, phase2_duration))

        traci.trafficlight.setProgramLogic(control_tl, tl_logic[0])

        reward_list = []

        with open("singleAgent-inout-action-phase.txt", "a") as f:
            f.write("SimulationStep {} action {} p1_duration {} p2_duration {}\n".format(simulationSteps, action, phase1_duration, phase2_duration))
        
        for i in range(90):
            traci.simulationStep()
            tpnumber += traci.simulation.getStartingTeleportNumber()
            simulationSteps += 1

            reward_tmp = -(traci.edge.getLastStepVehicleNumber(target_links[0]) - traci.edge.getLastStepVehicleNumber(target_links[1]))
            reward_list = np.append(reward_list, reward_tmp)
            arrived_arr.append(traci.simulation.getArrivedNumber())
            with open("singleAgent-inout-arrivednum.txt", "a") as f:
                f.write("Simulation {} arrived_num {}\n".format(simulationSteps, np.sum(arrived_arr)))
        
        total_reward += np.mean(reward_list)
        total_reward_withaction += np.mean(reward_list) + phase1_duration*0.1

        print('reward : {}'.format(np.mean(reward_list)))

    print('total reward : {}'.format(total_reward))
    print('total reward with action : {}'.format(total_reward_withaction))
    traci.close()