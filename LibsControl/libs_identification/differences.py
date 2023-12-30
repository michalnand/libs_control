


#https://en.wikipedia.org/wiki/Finite_difference_coefficient

def first_difference_1(x, dt):

    x_p0 = x[1:,    :]
    x_p1 = x[0:-1,  :]

    y = (1.0)*x_p0
    y+= (-1.0)*x_p1
   
    y = y/dt

    return y


def first_difference_4(x, dt):

    x_p0 = x[3:,    :]
    x_p1 = x[2:-1,  :]
    x_p2 = x[1:-2,  :]
    x_p3 = x[0:-3,  :]
    
    y = (11.0/6.0)*x_p0
    y+= (-3.0)*x_p1
    y+= (3.0/2.0)*x_p2
    y+= (-1.0/3.0)*x_p3

    y = y/dt 

    return y


def first_difference_6(x, dt):

    x_p0 = x[6:,    :]
    x_p1 = x[5:-1,  :]
    x_p2 = x[4:-2,  :]
    x_p3 = x[3:-3,  :]
    x_p4 = x[2:-4,  :]
    x_p5 = x[1:-5,  :]
    x_p6 = x[0:-6,  :]

    y = (49.0/20.0)*x_p0
    y+= (-6.0)*x_p1
    y+= (15.0/2.0)*x_p2
    y+= (-20.0/3.0)*x_p3
    y+= (15.0/4.0)*x_p4
    y+= (-6.0/5.0)*x_p5
    y+= (1.0/6.0)*x_p6

    y = y/dt 

    return y


def second_difference_1(x, dt):

    x_p0 = x[2:,    :]
    x_p1 = x[1:-1,  :]
    x_p2 = x[0:-2,  :]
    
    y = (1.0)*x_p0
    y+= (-2.0)*x_p1
    y+= (1.0)*x_p2
    
    y = y/(dt**2)

    return y

def second_difference_6(x, dt):
    x_p0 = x[7:,    :]
    x_p1 = x[6:-1,  :]
    x_p2 = x[5:-2,  :]
    x_p3 = x[4:-3,  :]
    x_p4 = x[3:-4,  :]
    x_p5 = x[2:-5,  :]
    x_p6 = x[1:-6,  :]
    x_p7 = x[0:-7,  :]

    y = (469.0/90.0)*x_p0
    y+= (-223.0/10.0)*x_p1
    y+= (879.0/20.0)*x_p2
    y+= (-949.0/18.0)*x_p3
    y+= (41.0)*x_p4
    y+= (-201.0/10.0)*x_p5
    y+= (1019.0/180.0)*x_p6
    y+= (-7.0/10.0)*x_p7

    y = y/dt 

    return y