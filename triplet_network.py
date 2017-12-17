
# coding: utf-8

# # Implémentation de Deep Learning Using Triplet Network
# 
# ## Première partie : Récupération des sets que nous allons utiliser
# 
# ### Cifar10
# 
# https://www.cs.toronto.edu/~kriz/cifar.html
# 
# ### MNIST
# 
# http://yann.lecun.com/exdb/mnist/
# 
# ### SVHN
# 
# http://ufldl.stanford.edu/housenumbers/
# 
# ### STL10
# 
# http://cs.stanford.edu/~acoates/stl10/
# 
# 
# On va commencer par uniquement Cifar10 dans un premier temps afin de simplifier la tâche.
# 
# Le Cifar10 est découpé en 5 parties train + la partie test.

# In[1]:

import tensorflow as tf
import random
import numpy
from numpy import linalg as LA

CIFAR10_PATH = r'C:\Users\leobo_000\Desktop\Developpement\TF_stage\cifar-10-python\cifar-10-batches-py'

batch_size_train = 20

batch_number = 20000

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# ## Etape 2 : Récupération sous Python
# 
# Le Cifar10 est découpé en 5 parties train + la partie test que nous allons récupéré.

# In[2]:

dic1 = unpickle(CIFAR10_PATH + '\\data_batch_1')
dic2 = unpickle(CIFAR10_PATH + '\\data_batch_2')
dic3 = unpickle(CIFAR10_PATH + '\\data_batch_3')
dic4 = unpickle(CIFAR10_PATH + '\\data_batch_4')
dic5 = unpickle(CIFAR10_PATH + '\\data_batch_5')
list_dic_train = [dic1, dic2, dic3, dic4, dic5]
dictest = unpickle(CIFAR10_PATH + '\\test_batch')


# ## Etape 3 : Création d'un set global, et d'un set par classe
# 
# Création d'un dictionnaire ayant pour clé la classe et comme valeur la liste des images

# In[3]:

gen_dic_train = ((i,[]) for i in range(0,10,1))
dic_train = dict(gen_dic_train)

i = 0

while i < len(dic1[b'labels']):
    for d in list_dic_train:
        key = d[b'labels'][i]
        data = d[b'data'][i]
        if key in dic_train:
            dic_train[key].append(data)
        else:
            dic_train[key] = [data]
    i=i+1


# ## Etape 4 : Création de la fonction permettant de créer les triplets pour le train
# 
# 640 000 triplets par epoch, générer aléatoirement à chaque epoch

# In[4]:

random.seed()

def creation_triplet(dic, nombre):
    triplet = [[],[],[]]
    i = 0
    while i<nombre:
        
        i+=1
        
        x_c = random.randrange(0, 10, 1)
        xm_c = random.randrange(0, 9, 1)
        x_n = random.randrange(0, 5000, 1)
        xp_n = random.randrange(0, 4999, 1)
        xm_n = random.randrange(0, 5000, 1)
        
        if(xm_c >= x_c):
            xm_c += 1;
        if(xp_n >= x_n):
            xp_n += 1;
        
        x = dic[x_c][x_n]
        xp = dic[x_c][xp_n]
        xm = dic[xm_c][xm_n]
        
        triplet[0].append(x)
        triplet[1].append(xp)
        triplet[2].append(xm)
        
    return triplet


# ## Etape 5 : Création des différents niveaux du réseau

# In[5]:

#On ne commence pas à 0 afin d'éviter les neurones mort
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# In[6]:

#Vecteur représentant la taille des images traitées
x = tf.placeholder(tf.float32, shape=[None, 3072])
xp = tf.placeholder(tf.float32, shape=[None, 3072])
xm = tf.placeholder(tf.float32, shape=[None, 3072])

#Représentant la valeur du dropout
keep_prob = tf.placeholder(tf.float32)

V_conv1 = weight_variable([5, 5, 3, 64])
V_conv2 = weight_variable([3, 3, 64, 128])
V_conv3 = weight_variable([3, 3, 128, 256])
V_conv4 = weight_variable([2, 2, 256, 128])

#Taille de l'image
x_image = tf.reshape(x, [-1, 32, 32, 3])
xp_image = tf.reshape(xp, [-1, 32, 32, 3])
xm_image = tf.reshape(xm, [-1, 32, 32, 3])


# In[7]:

def network(x_image):
    #Les 3 premiers CNN

    conv1 = tf.nn.relu(conv2d(x_image, V_conv1))
    conv2 = tf.nn.relu(conv2d(conv1, V_conv2))
    conv3 = tf.nn.relu(conv2d(conv2, V_conv3))

    #Le Max Pooling 2x2

    pool = tf.nn.relu(max_pool_2x2(conv3))

    #Le dernier CNN

    conv4 = conv2d(pool, V_conv4)
    
    #On rajoute le dropout pour éviter l'overfit (pas sûr de sa position en revanche)
    #conv4_drop = tf.nn.dropout(conv4, keep_prob)
    
    return conv4


# In[8]:

def euclidean_distance(vect1, vect2):
    return tf.sqrt(tf.reduce_sum(tf.square((tf.subtract(vect1, vect2)))))


# In[9]:

net_x = network(x_image)
net_xm = network(xm_image)
net_xp = network(xp_image)

soft_max_results = []
intermediary = []

for i in range(batch_size_train):
    dist_moins = euclidean_distance(tf.reshape(net_x[i],[-1]), tf.reshape(net_xm[i],[-1]))
    dist_plus = euclidean_distance(tf.reshape(net_x[i],[-1]), tf.reshape(net_xp[i],[-1]))
    intermediary.append([dist_moins, dist_plus])
    soft_max_results.append(tf.nn.softmax((dist_moins, dist_plus)))


# ## Création de la loss function
# 
# Erreur quadratique moyenne sur le soft-max du résultat

# In[10]:

loss_function = tf.losses.mean_squared_error([(0,1)]*batch_size_train, soft_max_results)


# ## Création de la fonction de train avec l'optimisation

# In[ ]:

learning_rate_start = 0.5
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(learning_rate_start, global_step,
                                           20, 0.8, staircase=True)

train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss_function
                                                                    , aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

correct_prediction = []

for i in range(batch_size_train):    
    correct_prediction.append(tf.equal(tf.argmax(soft_max_results[i]), tf.argmax(tf.constant([0,1]))))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# ## Réalisation du train
# 
# Ici, normalement on doit entraîner l'algorithme en fonction d'épochs composés de 640 000 triplets et l'on doit arrêter quand la précision stagne (normalement entre 10-30 épochs)

# In[ ]:

train_accuracy = 0
step_to_print = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(batch_number):
        batch = creation_triplet(dic_train, batch_size_train)
        train_step.run(feed_dict={x: batch[0], xp: batch[1], xm: batch[2], keep_prob: 0.5})
        
        train_accuracy += accuracy.eval(feed_dict={x: batch[0], xp: batch[1], xm: batch[2], keep_prob: 1})
        step_to_print += batch_size_train
        
        if(step_to_print >= 10000):
            number_of_step = step_to_print/batch_size_train
            print('step %d, %d triplets trained, training accuracy %g' % (i, (i+1)*batch_size_train, train_accuracy/number_of_step))
            train_accuracy = 0
            step_to_print = 0


# ## Réalisation de la fonction de perte multiclasses
# 
# Nous devons maintenant réaliser une fonction de perte permettant de classifier les images selon les 10 classes.

# In[ ]:



