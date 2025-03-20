import l2f
from .main import device, env, params, loop
ui = l2f.UI()
ui.ns = "l2f"
import websockets
import asyncio
import json
async def main():
    uri = "ws://localhost:13337/backend"
    async with websockets.connect(uri) as websocket:
        handshake = json.loads(await websocket.recv(uri))
        print(f"Handshake: {handshake}")
        assert(handshake["channel"] == "handshake")
        namespace = handshake["data"]["namespace"]
        print(f"Namespace: {namespace}")
        ui.ns = namespace
        ui_message = l2f.set_ui_message(device, env, ui)
        parameters_message = l2f.set_parameters_message(device, env, params, ui)

        await websocket.send(ui_message)
        await websocket.send(parameters_message)
        for step_i, state in enumerate(loop(N_STEPS=1000)):
            state_action_message = l2f.set_state_action_message(device, env, params, ui, state, [0, 0, 0, 0])
            await websocket.send(state_action_message)
            await asyncio.sleep(0.01)

if __name__ == "__main__":
    print("Please run:\ndocker run -it --rm -p 13337:13337 rltools/ui-server:2.0.0\nAnd open the UI in your browser at the displayed URL")
    asyncio.run(main())

