import numpy


def constrained_trajectory_following(x, max_vel, max_acc, max_jerk, robot, dt=0.01):
    N = len(x)
    executed = []
    
    # Initial desired state
    xr = x[0]
    vel = 0.0
    acc = 0.0
    jerk = 0.0
    
    for i in range(N):
        # Compute desired delta
        dx = x[i] - xr  
        
        # Desired position step based on current velocity
        # Limit jerk first
        desired_jerk = (dx / dt**3 - 3*acc/dt - 3*vel/dt**2)
        jerk += numpy.clip(desired_jerk - jerk, -max_jerk*dt, max_jerk*dt)
        jerk = numpy.clip(jerk, -max_jerk, max_jerk)
        
        acc += jerk * dt
        acc = numpy.clip(acc, -max_acc, max_acc)
        
        vel += acc * dt
        vel = numpy.clip(vel, -max_vel, max_vel)
        
        xr += vel * dt
        
        # Step the robot
        state = robot.step(xr)
        executed.append(state)
    
    return numpy.array(executed)