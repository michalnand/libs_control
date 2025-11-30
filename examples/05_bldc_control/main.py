import numpy
import matplotlib.pyplot as plt

from sine_table_lib import *

ENCODER_RESOLUTION  = 4096
MOTOR_POLES         = 14
    
PWM_VALUE_MAX       = 5000

SQRT3INV            = 591       #1/sqrt(3) = 591/1024

def set_torque_from_rotation(torque, rotor_angle):
    torque = numpy.clip(torque, -PWM_VALUE_MAX, PWM_VALUE_MAX)

    # convert mechanical angle to electrical angle
    pole_pairs  = MOTOR_POLES // 2
    elec_angle  = (rotor_angle * pole_pairs) % ENCODER_RESOLUTION

    table_angle = (elec_angle * SINE_TABLE_SIZE) // ENCODER_RESOLUTION

    if torque < 0:
        table_angle = (table_angle + SINE_TABLE_SIZE // 2) % SINE_TABLE_SIZE
        torque = -torque

    
    # produce 3-phase shifted angles
    angle_a = table_angle
    angle_b = (table_angle + SINE_TABLE_SIZE // 3) % SINE_TABLE_SIZE
    angle_c = (table_angle + 2 * (SINE_TABLE_SIZE // 3)) % SINE_TABLE_SIZE

    # read sinusoidal values
    sa = sine_table[angle_a]
    sb = sine_table[angle_b]
    sc = sine_table[angle_c]

    # center sinusoids around zero:
    sa -= SINE_VALUE_MAX//2
    sb -= SINE_VALUE_MAX//2
    sc -= SINE_VALUE_MAX//2     


    #transform into space-vector modulation, to achieve full voltage range
    min_val = min([sa, sb, sc])
    max_val = max([sa, sb, sc])

    com_val = (min_val + max_val)//2

    #normalise into 0..control_max
    a_norm = ( ((sa - com_val)*2*SQRT3INV)//1024 ) + SINE_VALUE_MAX//2
    b_norm = ( ((sb - com_val)*2*SQRT3INV)//1024 ) + SINE_VALUE_MAX//2
    c_norm = ( ((sc - com_val)*2*SQRT3INV)//1024 ) + SINE_VALUE_MAX//2

    #compute PWM value
    pwm_a = (a_norm*torque)//SINE_VALUE_MAX
    pwm_b = (b_norm*torque)//SINE_VALUE_MAX
    pwm_c = (c_norm*torque)//SINE_VALUE_MAX


    return elec_angle, [sa, sb, sc], [a_norm, b_norm, c_norm], [pwm_a, pwm_b, pwm_c]




if __name__ == "__main__":
    
    n_steps = ENCODER_RESOLUTION

    # one full rotation
    steps        = numpy.arange(n_steps, dtype=int)
    rotor_angles = numpy.arange(n_steps, dtype=int)%ENCODER_RESOLUTION
    
    elec_angle_result   = []
    sine_phases_result  = []
    f_values_result     = []
    pwm_values_result   = []

    for n in range(n_steps):
        torque = PWM_VALUE_MAX
       
        rotor_angle = rotor_angles[n]
        elec_angle, [sa, sb, sc], [fa, fb, fc], [pwm_a, pwm_b, pwm_c] = set_torque_from_rotation(torque, rotor_angle)

        elec_angle_result.append(elec_angle)
        sine_phases_result.append([sa, sb, sc])
        f_values_result.append([fa, fb, fc])
        pwm_values_result.append([pwm_a, pwm_b, pwm_c])

    elec_angle_result  = numpy.array(elec_angle_result)
    sine_phases_result = numpy.array(sine_phases_result)
    f_values_result    = numpy.array(f_values_result)
    pwm_values_result  = numpy.array(pwm_values_result)


    tmp = numpy.sum(pwm_values_result, axis=-1)
    print(tmp)

    print(numpy.max(pwm_values_result), numpy.min(pwm_values_result))


    fig, axs = plt.subplots(4, 1, figsize=(8, 2*4))


    axs[0].plot(steps, rotor_angles*360/ENCODER_RESOLUTION, label="mechanical angle", color="blueviolet")
    axs[0].plot(steps, elec_angle_result*360/ENCODER_RESOLUTION, label="electrical angle", color="skyblue")
    axs[0].set_xlabel("step")
    axs[0].set_ylabel("angle [degrees]")
    axs[0].legend()
    axs[0].grid()


    axs[1].plot(steps, sine_phases_result[:, 0], label="a", color="red")
    axs[1].plot(steps, sine_phases_result[:, 1], label="b", color="green")
    axs[1].plot(steps, sine_phases_result[:, 2], label="c", color="blue")
    axs[1].set_xlabel("step")
    axs[1].set_ylabel("phase")
    axs[1].legend()
    axs[1].grid()


    axs[2].plot(steps, f_values_result[:, 0], label="a", color="red")
    axs[2].plot(steps, f_values_result[:, 1], label="b", color="green")
    axs[2].plot(steps, f_values_result[:, 2], label="c", color="blue")
    axs[2].set_xlabel("step")
    axs[2].set_ylabel("space vector")
    axs[2].legend()
    axs[2].grid()


    axs[3].plot(steps, pwm_values_result[:, 0], label="a", color="red")
    axs[3].plot(steps, pwm_values_result[:, 1], label="b", color="green")
    axs[3].plot(steps, pwm_values_result[:, 2], label="c", color="blue")
    axs[3].set_xlabel("step")
    axs[3].set_ylabel("pwm")
    axs[3].legend()
    axs[3].grid()

    
    plt.tight_layout()
    plt.show()


    