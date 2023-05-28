import numpy

def const_augmentation(x):
    batch_size = x.shape[0]
    result   = numpy.ones((batch_size, 1), dtype=numpy.float32)

    return result


def polynomial_augmentation(x):
    batch_size      = x.shape[0]
    features_count  = x.shape[1]

    result          = numpy.zeros((batch_size,  features_count*features_count), dtype=numpy.float32)

    for j in range(features_count):
        for i in range(features_count):
            v       = x[:, j]*x[:, i]
            ptr =  j*features_count + i
            result[:, ptr] = v

    return result
            

def rotation_augmentation(x):
    batch_size      = x.shape[0]
    features_count  = x.shape[1]

    result          = numpy.zeros((batch_size,  2*features_count*features_count), dtype=numpy.float32)
    result[:, 0:features_count] = x

    for j in range(features_count):
        for i in range(features_count):
            ptr =  2*(j*features_count + i) + 0
            result[:, ptr] = x[:, j]*numpy.sin(x[:, i])

            ptr =  2*(j*features_count + i) + 1
            result[:, ptr] = x[:, j]*numpy.cos(x[:, i])

    return result
            
