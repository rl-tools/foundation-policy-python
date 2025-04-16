# A Foundation Model for Continuous Control


Supports Python >= `3.6`


```
from foundation_model import QuadrotorPolicy
import numpy as np
from scipy.spatial.transform import Rotation as R

policy = QuadrotorPolicy()
position = [0, 0, 0]
orientation = [1, 0, 0, 0]
orientation_rotation_matrix = R.from_quat(np.array(orientation)[[1, 2, 3, 0]]).as_matrix().ravel().tolist()
linear_velocity = [0, 0, 0]
angular_velocity = [0, 0, 0]
previous_action = [0, 0, 0, 0]

observation = np.array([position + orientation_rotation_matrix + linear_velocity + angular_velocity + previous_action])
policy.reset()
action = policy.evaluate_step(observation)
print(action)
```