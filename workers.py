# a ripoff of asyncsrv http://zguide.zeromq.org/py:all
# and http://pyzmq.readthedocs.io/en/latest/serialization.html

import zmq
import multiprocessing as mp
import numpy as np

class Client(mp.Process):
    def __init__(self, id, fxn):
        self.id = id
        self.fxn = fxn
        super(Client, self).__init__()

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        identity = u'worker-%d' % self.id
        socket.identity = identity.encode('ascii')
        socket.connect('tcp://localhost:5570')
        poll = zmq.Poller()
        poll.register(socket, zmq.POLLIN)
        reqs = 0
        while True:
            reqs = reqs + 1
            socket.send_string(u'request #%d' % (reqs))
            for i in range(5):
                sockets = dict(poll.poll(1000))
                if socket in sockets:
                    msg = socket.recv()

        socket.close()
        context.term()

class Router(mp.Process):
    """ServerTask"""
    def __init__(self, id, n_workers=1, front_port = 5570, back_port = 5580):
        super(Router, self).__init__()
        self.id = id
        self.n_workers=n_workers
        self.front_port = front_port
        self.back_port = back_port

    def run(self):
        context = zmq.Context()
        frontend = context.socket(zmq.ROUTER)
        frontend.bind('tcp://localhost:{}'.format(self.front_port))

        backend = context.socket(zmq.DEALER)
        backend.bind('tcp://localhost:{}'.format(self.back_port))

        workers = []
        for i in range(self.n_workers):
            worker = Worker(i, router_port = self.back_port)
            worker.start()
            workers.append(worker)

        zmq.proxy(frontend, backend)

        frontend.close()
        backend.close()
        context.term()

class Worker(mp.Process):
    """ServerWorker"""
    def __init__(self, id, fxn, router_port = 6680):
        super(Worker, self).__init__()
        self.id = id
        self.fxn = fxn
        self.router_port = router_port

    def run(self):
        self.context = zmq.Context()
        worker = self.context.socket(zmq.DEALER)
        worker.connect('tcp://localhost:{}'.format(self.router_port))
        while True:
            # a message will have kwargs and perhaps a numpy array
            # we first send the kwargs and then the array with a sendmore flag
            msg = worker.recv_json()
            array =




            # do work replies
            reply = self.fxn()
            for i in range(replies):
                worker.send_multipart([ident, msg])

        worker.close()


def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = buffer(msg)
    A = np.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])