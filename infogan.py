import numpy as np
import matplotlib as mpl
mpl.use('PDF')
import matplotlib.pyplot as plt

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class GenMLP(Chain):
    def __init__(self, dim_z):
        super(GenMLP, self).__init__(
            ll1=L.Linear(dim_z, 1024),
            bn1=L.BatchNormalization(1024),
            ll2=L.Linear(1024, 7*7*128),
            bn2=L.BatchNormalization(7*7*128),
            dc3=L.Deconvolution2D(128, 64, 4, stride=2, pad=1),
            bn3=L.BatchNormalization(64),
            dc4=L.Deconvolution2D(64, 1, 4, stride=2, pad=1),
        )

    def __call__(self, z, test=False):
        h = F.relu(self.bn1(self.ll1(z), test=test))
        h = F.relu(self.bn2(self.ll2(h), test=test))
        h = F.reshape(h, (z.shape[0], 128, 7, 7,))
        h = F.relu(self.bn3(self.dc3(h), test=test))
        return self.dc4(h)


class DisMLP(Chain):
    def __init__(self, dim_cat, dim_cont):
        super(DisMLP, self).__init__(
            cl1=L.Convolution2D(1, 64, 4, stride=2, pad=1),
            cl2=L.Convolution2D(64, 128, 4, stride=2, pad=1),
            bn2=L.BatchNormalization(128),
            ll3=L.Linear(7*7*128, 1024),
            bn3=L.BatchNormalization(1024),

            d_fc=L.Linear(1024, 1),

            q_fc1=L.Linear(1024, 128),
            q_bn=L.BatchNormalization(128),
            q_fc2=L.Linear(128, dim_cat+dim_cont*2),
        )
        self.dim_cat = dim_cat
        self.dim_cont = dim_cont

    def __call__(self, x, test=False):
        h = F.leaky_relu(self.cl1(x), slope=0.1)
        h = F.leaky_relu(self.bn2(self.cl2(h), test=test), slope=0.1)
        h = F.leaky_relu(self.bn3(self.ll3(h), test=test), slope=0.1)
        return self.d_fc(h)

    def discriminator(self, x, test=False):
        h = F.leaky_relu(self.cl1(x), slope=0.1)
        h = F.leaky_relu(self.bn2(self.cl2(h), test=test), slope=0.1)
        h = F.leaky_relu(self.bn3(self.ll3(h), test=test), slope=0.1)
        return self.d_fc(h)

    def recognizer(self, x, test=False):
        h = F.leaky_relu(self.cl1(x), slope=0.1)
        h = F.leaky_relu(self.bn2(self.cl2(h), test=test), slope=0.1)
        h = F.leaky_relu(self.bn3(self.ll3(h), test=test), slope=0.1)

        qh = F.leaky_relu(self.q_bn(self.q_fc1(h), test=test), slope=0.1)
        return self.q_fc2(qh), self.d_fc(h)


class InfoGAN:
    def __init__(self, generator, dis_recog,
                 dim_data, dim_z, dim_cat, dim_cont
    ):
        self.generator = generator
        self.dis_recog = dis_recog
        self.discriminator = dis_recog.discriminator
        self.recognizer = dis_recog.recognizer
        self.dim_data = dim_data
        self.dim_z = dim_z
        self.dim_cat = dim_cat
        self.dim_cont = dim_cont
        self.setup_optimizers()

    def dis_loss(self, x, t):
        return F.sigmoid_cross_entropy(self.discriminator(x), t)

    def gen_loss(self, noise, c_cat, c_cont):
        cat_onehot = chainer.Variable(
            np.eye(self.dim_cat).astype(np.float32).take(c_cat.data, axis=0)
        )
        z = F.hstack((noise, cat_onehot, c_cont,))

        x = self.generator(z)
        rec, dis = self.recognizer(x)

        g_loss = F.sigmoid_cross_entropy(
            dis, chainer.Variable(np.ones((z.shape[0], 1), dtype=np.int32))
        )

        r_cat, r_mean, r_logvar = F.split_axis(
            rec, [self.dim_cat, self.dim_cat+self.dim_cont], 1
        )
        m_info = F.gaussian_nll(c_cont, r_mean, r_logvar) / z.shape[0]
        m_info = m_info + F.softmax_cross_entropy(
            r_cat, c_cat, normalize=False
        )

        return g_loss, m_info

    def setup_optimizers(self):
        self.gen_optimizer = optimizers.Adam(alpha=1e-3, beta1=0.5)
        self.gen_optimizer.use_cleargrads()
        self.gen_optimizer.setup(self.generator)
        self.dis_optimizer = optimizers.Adam(alpha=2e-4, beta1=0.5)
        self.dis_optimizer.use_cleargrads()
        self.dis_optimizer.setup(self.dis_recog)

    def train(self, dataset, n_epochs, batch_size, report=None, z_test=None):
        label0s = chainer.Variable(np.zeros((batch_size, 1), dtype=np.int32))
        label1s = chainer.Variable(np.ones((batch_size, 1), dtype=np.int32))
        # Epochs
        for epoch in range(1, n_epochs+1):
            print("Initiating an epoch #", epoch, flush=True)
            perm = np.random.permutation(dataset.shape[0])

            dis_epochloss = 0.0
            gen_epochloss = 0.0

            for i in range(0, dataset.shape[0], batch_size):
                # Process a mini batch
                data_batch = chainer.Variable(dataset[perm[i:i+batch_size]])

                # Update discriminator
                z_batch = chainer.Variable(self.sample_z(batch_size))

                dis_batchloss = self.dis_loss(data_batch, label1s)
                dis_batchloss = dis_batchloss + self.dis_loss(
                    self.generator(z_batch), label0s
                )

                # Update generator
                noise_batch = chainer.Variable(self.sample_noise(batch_size))
                cat_batch = chainer.Variable(self.sample_cat(batch_size))
                cont_batch = chainer.Variable(self.sample_cont(batch_size))

                gen_batchloss, mut_info = self.gen_loss(
                    noise_batch, cat_batch, cont_batch
                )
                gen_batchloss = gen_batchloss + mut_info
                dis_batchloss = dis_batchloss + mut_info

                self.generator.cleargrads()
                gen_batchloss.backward()
                self.gen_optimizer.update()

                self.dis_recog.cleargrads()
                dis_batchloss.backward()
                self.dis_optimizer.update()

                dis_epochloss = dis_epochloss + dis_batchloss.data
                gen_epochloss = gen_epochloss + gen_batchloss.data

            print("    Discriminator loss in a epoch :", dis_epochloss)
            print("    Generator loss in a epoch     :", gen_epochloss)
            print("    Mutual Information            :", mut_info.data)
            print("", flush=True)
            if (report is not None) and (z_test is not None):
                report(self.generator, self.discriminator, self.recognizer,
                       epoch, z_test
                )

    def sample_z(self, n_samples):
        z_batch = self.sample_noise(n_samples)
        cat_batch = np.eye(self.dim_cat).astype(np.float32).take(
            self.sample_cat(n_samples), axis=0
        )
        cont_batch = self.sample_cont(n_samples)

        return np.column_stack((z_batch, cat_batch, cont_batch,))

    def sample_noise(self, n_samples):
        return np.random.uniform(
            -1, 1, (n_samples, self.dim_z)
        ).astype(np.float32)

    def sample_cat(self, n_samples):
        return np.random.randint(self.dim_cat, size=n_samples).astype(np.int32)

    def sample_cont(self, n_samples):
        return np.random.uniform(
            -1, 1, (n_samples, self.dim_cont)
        ).astype(np.float32)


def make_dataset():
    # MNIST dataset without labels
    return datasets.get_mnist(withlabel=False, ndim=3)

# def update_net(loss, optimizer, arg1, arg2):
#     loss.cleargrads()
#     batchloss = loss(arg1, arg2)
#     batchloss.backward()
#     optimizer.update()
#     return batchloss

def save_digits(generator, discriminator, recognizer, epoch, z_test):
    fig = plt.figure()
    samples = generator(z_test).data
    samples = [sample[0] for sample in samples]
    for i, sample in zip(range(1, 100+1), samples):
        ax = fig.add_subplot(10, 10, i)
        ax.matshow(sample, cmap=mpl.cm.gray)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
#    plt.show()
    plt.savefig("{:04d}.pdf".format(epoch))


def main():
    dim_data = (28, 28)

    # Hyperparameters
#    dim_z = 10
#    gen_hidden, dis_hidden = 5, 5
    dim_z = 62
    dim_cat = 10
    dim_cont = 2

    batch_size = 100
    n_epochs = 1000

    generator = GenMLP(dim_z+dim_cat+dim_cont)
    discriminator = DisMLP(dim_cat, dim_cont)

    gans = InfoGAN(
        generator, discriminator,
        dim_data, dim_z, dim_cat, dim_cont
    )

    train_data, test_data = make_dataset()

    noise_batch = np.random.uniform(
        -1, 1, (100, dim_z)
    ).astype(np.float32)
    cat_batch = np.vstack(
        (np.eye(10).astype(np.float32) for _ in range(10))
    )
    cont_batch = np.random.uniform(
        -2, 2, (100, dim_cont)
    ).astype(np.float32)
    z_batch = np.hstack((noise_batch, cat_batch, cont_batch,))

    gans.train(
        train_data, n_epochs, batch_size,
        report=save_digits, z_test=chainer.Variable(z_batch)
    )

    serializers.save_npz("generator.npz", gans.generator)
    serializers.save_npz("discriminator.npz", gans.discriminator)

if __name__ == '__main__':
    main()
