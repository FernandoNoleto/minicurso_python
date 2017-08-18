import numpy, time, theano
import theano.tensor as t


a = numpy.matrix ( [[1,2], [3,4]] )
b = numpy.matrix ( [[5,6], [7, 8]] )
c = numpy.dot(a, b)
print c

#-------------------------------------------------------

a = numpy.random.uniform(-10, 10, size=(50, 500))
b = numpy.random.uniform(-10, 10, size=(500, 10))
t1 = time.time()
c = numpy.dot(a, b)
t2 = time.time() - t1
print c
print t2

#-------------------------------------------------------

m1 = t.dmatrix('m1')
m2 = t.dmatrix('m2')

f = theano.function([m1, m2], t.dot(m1, m2))
t3 = time.time()
c = f(a,b)
t4 = time.time() - t3
print t4