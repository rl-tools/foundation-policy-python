from foundation_model import load_quadrotor
import numpy as np
from scipy.spatial.transform import Rotation as R
import l2f
import os
from  copy import copy

quadrotor = load_quadrotor()



position = [0, 0, 0]
orientation = [1, 0, 0, 0]
orientation_rotation_matrix = R.from_quat(np.array(orientation)[[1, 2, 3, 0]]).as_matrix().ravel().tolist()
linear_velocity = [0, 0, 0]
angular_velocity = [0, 0, 0]
previous_action = [0, 0, 0, 0]

observation = np.array([position + orientation_rotation_matrix + linear_velocity + angular_velocity + previous_action])
quadrotor.reset()
action = quadrotor.evaluate_step(observation)
print(action)



device = l2f.Device()

rng = l2f.Rng()

env = l2f.Environment()
params = l2f.Parameters()
state = l2f.State()
observation = l2f.Observation()
next_state = l2f.State()
observation = l2f.Observation()
l2f.initialize_environment(device, env)
l2f.initialize_rng(device, rng, 0)






ui = l2f.UI()
ui.ns = "l2f"

def loop(N_STEPS = 500):
    l2f.sample_initial_parameters(device, env, params, rng)
    l2f.initial_state(device, env, params, state)
    quadrotor.reset()
    for step_i in range(N_STEPS):
        l2f.observe(device, env, params, state, observation, rng)
        obs = np.concatenate([np.array(observation.observation)[:18], np.array(observation.observation)[-4:]]) # concatenating position, orientation (rotation matrix), linear velocity, angular velocity, and previous action (note in newer versions of l2f the most recent action follows right after the angular velocity)
        action = quadrotor.evaluate_step(np.array([obs]))[0]
        print("step: ", step_i, " position", state.position, " orientation", state.orientation, " linear_velocity", state.linear_velocity, " angular_velocity", state.angular_velocity, " rpm", state.rpm)
        l2f.step(device, env, params, state, action, next_state, rng)
        state.assign(next_state)
        yield copy(state)

for _ in loop():
    pass


# async def main():
#     uri = "ws://localhost:8080/backend"
#     async with websockets.connect(uri) as websocket:
#         handshake = json.loads(await websocket.recv(uri))
#         print(f"Handshake: {handshake}")
#         assert(handshake["channel"] == "handshake")
#         namespace = handshake["data"]["namespace"]
#         print(f"Namespace: {namespace}")
#         ui.ns = namespace
#         ui_message = set_ui_message(device, env, ui)
#         parameters_message = set_parameters_message(device, env, params, ui)

#         await websocket.send(ui_message)
#         await websocket.send(parameters_message)
#         for step_i in range(500):
#             # sleep for 1 second

#             step(device, env, params, state, [1, 0, 0, 0], next_state, rng)
#             state.assign(next_state)
#             state_action_message = set_state_action_message(device, env, params, ui, state, [0, 0, 0, 0])
#             await websocket.send(state_action_message)
#             await asyncio.sleep(0.1)




# if __name__ == "__main__":
#     asyncio.run(main())


