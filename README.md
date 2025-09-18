# A Foundation Policy for Quadrotor Control

Check out [rl-tools/raptor](https://github.com/rl-tools/raptor) for more details.

Supports Python >= `3.6`

## Usage
If you want to use your own simulator:
```bash
pip install foundation-policy
```
```python
from foundation_policy import Raptor
policy = Raptor()
policy.reset()
for simulation_step in range(1000):
    observation = np.array([[*sim.position, *R(sim.orientation).flatten(), *sim.linear_velocity, *sim.angular_velocity, *sim.action]])
    action = policy.evaluate_step(observation)[0] # the policy works on batches by default
    simulation.step(action) # simulation dt=10 ms
```
Note that the axis conventions are FLU (x = forward, y = left, z = up). Please convert position, orientation, linear velocity and angular velocity into these conventions. The angular velocity is in the body frame. The action motor conventions are [front-right, back-right, back-left, front-left] and the motor commands are normalized in the range [-1, 1].  


### Usage: L2F
The following instructions show how to use [l2f](https://github.com/rl-tools/l2f), the simulator used for training the foundation policy:
```bash
pip install l2f ui-server foundation-policy
```
Run `ui-server` in background and open [http://localhost:13337](http://localhost:13337)
```bash
ui-server
```
Then run the following code:
```python
from copy import copy
import numpy as np
import asyncio, websockets, json
import l2f
from l2f import vector8 as vector
from foundation_policy import Raptor

policy = Raptor()
device = l2f.Device()
rng = vector.VectorRng()
env = vector.VectorEnvironment()
ui = l2f.UI()
params = vector.VectorParameters()
state = vector.VectorState()
observation = np.zeros((env.N_ENVIRONMENTS, env.OBSERVATION_DIM), dtype=np.float32)
next_state = vector.VectorState()

vector.initialize_rng(device, rng, 0)
vector.initialize_environment(device, env)
vector.sample_initial_parameters(device, env, params, rng)
vector.sample_initial_state(device, env, params, state, rng)

async def render(websocket, state, action):
    ui_state = copy(state)
    for i, s in enumerate(ui_state.states):
        s.position[0] += i * 0.1 # Spacing for visualization
    state_action_message = vector.set_state_action_message(device, env, params, ui, ui_state, action)
    await websocket.send(state_action_message)

async def main():
    uri = "ws://localhost:13337/backend" # connection to the UI server
    async with websockets.connect(uri) as websocket:
        handshake = json.loads(await websocket.recv(uri))
        assert(handshake["channel"] == "handshake")
        namespace = handshake["data"]["namespace"]
        ui.ns = namespace
        ui_message = vector.set_ui_message(device, env, ui)
        parameters_message = vector.set_parameters_message(device, env, params, ui)
        await websocket.send(ui_message)
        await websocket.send(parameters_message)
        await asyncio.sleep(1)
        await render(websocket, state, np.zeros((8, 4)))
        await asyncio.sleep(2)
        policy.reset()
        for _ in range(500):
            vector.observe(device, env, params, state, observation, rng)
            action = policy.evaluate_step(observation[:, :22])
            dts = vector.step(device, env, params, state, action, next_state, rng)
            state.assign(next_state)
            await render(websocket, state, action)
            await asyncio.sleep(dts[-1])

if __name__ == "__main__":
    asyncio.run(main())
```

