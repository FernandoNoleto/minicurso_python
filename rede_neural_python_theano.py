import numpy, time, theano
import theano.tensor as t


'''

a = numpy.matrix ( [[1,2], [3,4]] )
b = numpy.matrix ( [[5,6], [7, 8]] )
c = numpy.dot(a, b)
print c

#--------------------------------------------------------------

a = numpy.random.uniform(-10, 10, size=(50, 500))
b = numpy.random.uniform(-10, 10, size=(500, 10))
t1 = time.time()
c = numpy.dot(a, b)
t2 = time.time() - t1
print c
print t2

#-------------------------------------------------------------

m1 = t.dmatrix('m1')
m2 = t.dmatrix('m2')

f = theano.function([m1, m2], t.dot(m1, m2))
t3 = time.time()
c = f(a,b)
t4 = time.time() - t3
print t4


#-------------------------------------------------------------

x = t.dscalar('x')
y = t.dscalar('y')
z = 0.26 * (t.pow(x,2) + t.pow(y, 2))-0.48*x*y
f = theano.function ([x, y], z)
grad = theano.function([x, y], t.grad(z, [x, y]))   #([x, y], t.grad(z, y, x))

valor1 = 0.2
valor2 = 0.0

while(1):
    g = grad(valor1, valor2)
    valor1 -= 0.001*grad(valor1, valor2)[0]
    valor2 -= 0.001*grad(valor1, valor2)[1]
    if numpy.linalg.norm(g) < 0.000001:
        break
print valor1, valor2, f(valor1, valor2)

#------------------------------------------------------------
'''

entrada_rede = t.dvector('in')
pesos_camada_oculta = theano.shared (value = numpy.random.uniform (-1/6, 1/6, size = (2 ,2)))

bias_camada_oculta = theano.shared(value = numpy.random.uniform (-1/6, 1/6, size = (2, )))

camada_oculta = t.tanh(t.dot(entrada_rede, pesos_camada_oculta + bias_camada_oculta))

pesos_camada_saida = theano.shared(value = numpy.random.uniform (-1/6, 1/6, size = (2, 1)))

bias_camada_saida = theano.shared(value = numpy.random.uniform (-1/6, 1/6, size = (1, )))

camada_saida = t.tanh(t.dot(camada_oculta, pesos_camada_saida) + bias_camada_saida)

propagar = theano.function([entrada_rede], camada_saida)
print propagar([1.0, 1.0])

pesos = [pesos_camada_oculta, bias_camada_oculta, pesos_camada_saida, bias_camada_saida]
saida_esperada = t.dvector('esperada')
erro = t.sum(t.pow(camada_saida - saida_esperada, 2))
gradiente = t.grad(erro, pesos)
lista_atualizacao = [(p, p - 0.01 * g) for p, g in zip(pesos, gradiente)]

treinar = theano.function ([entrada_rede, saida_esperada], erro, updates = lista_atualizacao)

print propagar ([-1.0, -1.0]), propagar ([-1.0, 1.0]), propagar ([1.0, -1.0]), propagar ([1.0, 1.0])

for i in range(100000):
    treinar([-1.0, -1.0], [-1.0])
    treinar([-1.0, 1.0], [1.0])
    treinar([1.0, -1.0], [1.0])
    treinar([1.0, 1.0], [1.0])
print propagar([-1.0, -1.0])
print propagar([-1.0, 1.0])
print propagar([1.0, -1.0])
print propagar([1.0, 1.0])
