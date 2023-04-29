# libs_control

# TODO 
- embedded LibsControlEmbedded
- working live aplication in : [brushless motor control](https://github.com/michalnand/brushless_motor_control)

# [balancing robot](examples/07_balancing_robot/)

- classical inverted pendulum problem
- goal is to control angle and position

 
![diagram](doc/images/balacing_robot.gif)

![diagram](examples/07_balancing_robot/results/closed_loop_response_observed.png)

![diagram](examples/07_balancing_robot/results/poles.png)

![diagram](examples/07_balancing_robot/results/poles_mesh_cl.png)


# [spring wheels](examples/06_lqri_wheels/)

- this great challenging test is taken from 
[www.do-mpc.com](https://www.do-mpc.com/en/latest/getting_started.html#Example-system)
- goal is to control position of wheels

![diagram](doc/images/wheels.gif)

![diagram](examples/06_lqri_wheels/results/closed_loop_response.png)

![diagram](examples/06_lqri_wheels/results/poles.png)

![diagram](examples/06_lqri_wheels/results/poles_mesh_cl.png)


 

# LQR controller

![diagram](doc/diagrams/control-lqr.png)

# LQR with integral action controller

- completly removes any constant disturbance

![diagram](doc/diagrams/control-lqri.png)

# LQG controller

- dont require observing fully state

![diagram](doc/diagrams/control-lqg.png)

